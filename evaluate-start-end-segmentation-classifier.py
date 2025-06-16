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

idx_to_labels = {
    # -1: ["transition"],
    0: ['walk'],
    1: ['hit', 'punch'],
    2: ['kick'],
    3: ['run', 'jog'],
    4: ['jump', 'hop', 'leap'],
    5: ['throw'],
    6: ['catch'],
    7: ['step'],
    # NOTE: large diversity expected
    8: ['greet'],
    9: ['dance'],
    # NOTE: large diversity expected
    10: ['stretch', 'yoga', 'exercise / training'],
    # NOTE: is this distinct enough from take / pick something up? Yeah, I think so.
    11: ['turn', 'spin'],
    12: ['bend'],
    # NOTE: large diversity expected
    13: ['stand'],
    14: ['sit'],
    15: ['kneel'],
    16: ['place something'],
    17: ['grasp object'],
    18: ['take/pick something up', 'lift something'],
    19: ['scratch', 'touching face', 'touching body parts'],
}
idx_to_labels = { key: ",".join(value) for key, value in idx_to_labels.items() }
labels_to_idx = { v: k for k, v in idx_to_labels.items() }
idx_to_labels = labels_to_idx

@hydra.main(version_base=None, config_path="configs", config_name="evaluate-start-end-segmentation-classifier")
def evaluate_start_end_segmentation_classifier(newcfg: DictConfig) -> None:
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, classification_report, confusion_matrix

    device = newcfg.device
    run_dir = newcfg.run_dir
    ckpt_name = newcfg.ckpt

    save_dir = os.path.join(run_dir, "classification-evaluation")
    
    os.makedirs(save_dir, exist_ok=True)

    cfg = read_config(run_dir)
    pl.seed_everything(cfg.seed)
    
    print("[encoder]:", cfg.model.motion_encoder)
    print("[classifier]:", cfg.model.classifier)

    model = load_model_from_cfg(
        cfg,
        ckpt_name,
        eval_mode=True,
        device=device,
        pretrained=True
    )
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

    all_predictions, all_groundtruths = [], []
    class_losses, start_losses, end_losses = [], [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="[evaluate-segmentation]"):
            batch["motion"] = batch["motion"].to(device)
            batch["transformed_motion"] = batch["transformed_motion"].to(device)
            batch["annotation"] = batch["annotation"].to(device)
                
            label = model.label_extractor.extract(batch["annotation"])
            
            class_logits, start_logits, end_logits = model.forward(batch, None)
            
            logger.debug("[batch.annotation]:", batch["annotation"].shape)
            logger.debug("[label.shape]:", label.shape)
            
            start_logits = start_logits.view(-1)
            end_logits = end_logits.view(-1)
            
            # NOTE: [batch, #classes]
            logger.debug("[class_logits.shape]:", class_logits.shape)
            # NOTE: [batch]
            logger.debug("[start_logits.shape]:", start_logits.shape)
            # NOTE: [batch]
            logger.debug("[end_logits.shape]:", end_logits.shape)
            
            loss, (class_loss, start_loss, end_loss) = model.compute_loss(class_logits, start_logits, end_logits, label)

            class_losses.append(class_loss.item())
            start_losses.append(start_loss.item())
            end_losses.append(end_loss.item())

            # NOTE: [batch]
            prediction = class_logits.argmax(dim=-1)
            groundtruth = label[:, 0]
            
            # NOTE: we do this to not compute the loss related to negative classes (not selected ones)
            valid_mask = label[:, 0] >= 0
            
            all_predictions.extend(prediction[valid_mask].long().cpu().tolist())
            all_groundtruths.extend(groundtruth[valid_mask].long().cpu().tolist())

    accuracy = accuracy_score(all_groundtruths, all_predictions)
    class_report = classification_report(all_groundtruths, all_predictions, target_names=idx_to_labels, output_dict=True)
    confusion = confusion_matrix(all_groundtruths, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_groundtruths, all_predictions)
    mcc = matthews_corrcoef(all_groundtruths, all_predictions)
    
    confusion_dataframe = pd.DataFrame(confusion, index=idx_to_labels)

    print(f"[balanced accuracy]: {balanced_accuracy * 100:.2f}%")
    print(f"[mcc]: {mcc}")
    print(f"[overall accuracy]: {accuracy * 100:.2f}%")
    print(f"--- --- ---")
    print("[per class accuracy]:")
    for cls, metrics in class_report.items():
        if cls in idx_to_labels:
            print(f"\t[class-{cls}]: {metrics['recall'] * 100:.2f}%")

    print(f"--- --- ---")
    
    print("[class-loss]:", sum(class_losses) / len(class_losses) if class_losses else "No classification loss computed")
    print("[start-loss]:", sum(start_losses) / len(start_losses) if start_losses else "No valid start targets")
    print("[end-loss]:", sum(end_losses) / len(end_losses) if end_losses else "No valid end targets")

    print(f"--- --- ---")
    
    print("[confusion matrix]:")
    print(confusion_dataframe)
    
    print(f"--- --- ---")

    print("[full classification report]:")
    print(classification_report(all_groundtruths, all_predictions, target_names=idx_to_labels))
    
    evaluation_metrics = {
        "overall_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "per_class_accuracy": {
            cls: metrics['recall'] for cls, metrics in class_report.items() if cls in idx_to_labels
        },
        "per_class_f1_score": {
            cls: metrics['f1-score'] for cls, metrics in class_report.items() if cls in idx_to_labels
        },
        "per_class_precision": {
            cls: metrics['precision'] for cls, metrics in class_report.items() if cls in idx_to_labels
        },
        "per_class_support": {
            cls: metrics['support'] for cls, metrics in class_report.items() if cls in idx_to_labels
        },
        "per_class_labels": idx_to_labels,
        "num_samples": len(all_groundtruths),
        "num_classes": len(idx_to_labels),
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
    
    print(f"--- --- ---")
    
    print(f"[metrics-saved-to]: {metrics_path}")

if __name__ == "__main__":
    evaluate_start_end_segmentation_classifier()
