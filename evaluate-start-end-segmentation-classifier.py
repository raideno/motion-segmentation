# HYDRA_FULL_ERROR=1 python evaluate-start-end-segmentation-segmenter.py --multirun \
#     data.dir=/home/nadir/disk/datasets/babel-for-validation \
#     run_dir=/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_10,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_15,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_20,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_25,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_30,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_10,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_15,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_20,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_25,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_30,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_10,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_15,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_20,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_25,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_30

import os
import tqdm
import yaml
import torch
import hydra
import logging
import src.prepare 

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from omegaconf import DictConfig
from src.config import read_config
from hydra.utils import instantiate
from src.logging import save_metric
from src.load import load_model_from_cfg

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="evaluate-start-end-segmentation-classifier")
def evaluate_start_end_segmentation_classifier(newcfg: DictConfig) -> None:
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    device = newcfg.device
    run_dir = newcfg.run_dir
    ckpt_name = newcfg.ckpt

    save_dir = os.path.join(run_dir, "classification-evaluation")
    
    os.makedirs(save_dir, exist_ok=True)

    cfg = read_config(run_dir)
    pl.seed_everything(cfg.seed)

    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    model.eval()

    dataset = instantiate(
        cfg.data,
        split="test"
    )
    dataloader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        # collate_fn=dataset.collate_fn,
        shuffle=False
    )

    all_preds, all_labels = [], []
    class_losses, start_losses, end_losses = [], [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="[evaluate-segmentation]"):
            for key in ["transformed_motion", "motion", "annotation"]:
                batch[key] = batch[key].to(device)
                
            label = model.label_extractor.extract(batch["annotation"])
            class_logits, start_logits, end_logits = model(batch, None)
            
            class_logits = class_logits.view(-1)
            start_logits = start_logits.view(-1)
            end_logits = end_logits.view(-1)
            
            loss, (class_loss, start_loss, end_loss) = model.compute_loss(class_logits, start_logits, end_logits, label)

            class_losses.append(class_loss.item())
            start_losses.append(start_loss.item())
            end_losses.append(end_loss.item())

            # NOTE: classification
            class_preds = (torch.sigmoid(class_logits.view(-1)) > 0.5).long()
            all_preds.extend(class_preds.cpu().tolist())
            all_labels.extend(label[:, 0].long().cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=["no_transition", "transition"], output_dict=True)
    confusion = confusion_matrix(all_labels, all_preds)
    confusion_dataframe = pd.DataFrame(confusion, index=["no_transition", "transition"], columns=["pred_no_transition", "pred_transition"])

    print(f"[overall accuracy]: {accuracy * 100:.2f}%")
    # for cls, metrics in class_report.items():
    #     if cls in ["0", "1"]:
    #         print(f"Class {cls} accuracy: {metrics['recall'] * 100:.2f}%")
    print("\n[per class accuracy]:")
    for cls, metrics in class_report.items():
        if cls in ["no_transition", "transition"]:
            print(f"Class '{cls}' accuracy: {metrics['recall'] * 100:.2f}%")

    print()
    print("[class-loss]:", sum(class_losses) / len(class_losses) if class_losses else "No classification loss computed")
    print("[start-loss]:", sum(start_losses) / len(start_losses) if start_losses else "No valid start targets")
    print("[end-loss]:", sum(end_losses) / len(end_losses) if end_losses else "No valid end targets")

    print("\n[confusion matrix]:")
    print(confusion_dataframe)

    print("\n[full classification report]:")
    print(classification_report(all_labels, all_preds, target_names=["no_transition", "transition"]))
    
    evaluation_metrics = {
        "overall_accuracy": accuracy,
        "per_class_accuracy": {
            "no_transition": class_report["no_transition"]["recall"],
            "transition": class_report["transition"]["recall"],
        },
        "losses": {
            "classification_loss": sum(class_losses) / len(class_losses) if class_losses else None,
            "start_loss": sum(start_losses) / len(start_losses) if start_losses else None,
            "end_loss": sum(end_losses) / len(end_losses) if end_losses else None,
        },
        "confusion_matrix": confusion.tolist(),
        "classification_report": class_report
    }

    metrics_path = os.path.join(save_dir, "metrics")
    
    save_metric(metrics_path, evaluation_metrics, format="yaml")
    save_metric(metrics_path, evaluation_metrics, format="json")
    
    print(f"\n[metrics-saved-to]: {metrics_path}")

if __name__ == "__main__":
    evaluate_start_end_segmentation_classifier()
