import os
import tqdm
import yaml
import torch
import hydra
import logging
import src.prepare  # noqa

import numpy as np
import pytorch_lightning as pl

from omegaconf import DictConfig
from src.config import read_config
from hydra.utils import instantiate
from src.load import load_model_from_cfg

logger = logging.getLogger(__name__)

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)

@hydra.main(version_base=None, config_path="configs", config_name="evaluate-classifier-segmentation")
def evaluate_classifier_segmentation(newcfg: DictConfig) -> None:
    device = newcfg.device
    run_dir = newcfg.run_dir
    examples = newcfg.examples
    ckpt_name = newcfg.ckpt

    save_dir = os.path.join(run_dir, "segmentation-evaluation")
    os.makedirs(save_dir, exist_ok=True)

    # NOTE: moved up here in order to use the segmentation config for the dataset
    # defined in the config file rather than the config used to train the model
    dataset = instantiate(newcfg.data, split="test")

    # NOTE: will load the config used to train the model
    cfg = read_config(run_dir)

    pl.seed_everything(cfg.seed)

    logger.info("[model]: loading")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    
    logger.info(f"[dataset.mode]: {dataset.mode}")
    
    dataloader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    model = model.eval()

    # --- --- --- ---

    all_preds, all_labels = [], []

    with torch.no_grad():
        for index, batch in tqdm.tqdm(iterable=enumerate(dataloader), total=len(dataloader), desc="[evaluate-segmentation]"):
            batch["transition_mask"] = batch["transition_mask"].to(device)
            
            batch["motion_x_dict"]["x"] = batch["motion_x_dict"]["x"].to(device)
            batch["motion_x_dict"]["mask"] = batch["motion_x_dict"]["mask"].to(device)
            
            x = batch["motion_x_dict"]
            y = batch["transition_mask"]
            
            outputs = model.segment_sequence(
                motion_x_dict=x,
                window_size=20,
                window_step=5
            )
            
            preds = outputs.cpu().numpy()
            labels = y.squeeze(1).cpu().numpy()
            
            for pred_seq, label_seq in zip(preds, labels):
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    acc_list = []
    edit_list = []
    
    f1_thresholds = np.arange(0.1, 1.0, 0.1)
    f1_scores = []
    
    from src.model.metrics import accuracy_score, levenshtein, f_score

    for pred_seq, label_seq in zip(all_preds, all_labels):
        acc = accuracy_score(label_seq, pred_seq)
        edit = levenshtein(pred_seq, label_seq)
        
        f1_thresholds_scores = []
        
        for threshold in f1_thresholds:
            f1_thresholds_scores.append(f_score(pred_seq, label_seq, overlap=threshold))

        acc_list.append(acc)
        edit_list.append(edit)
        f1_scores.append(f1_thresholds_scores)

    print("Frame-wise Accuracy: {:.2f}%".format(100 * np.mean(acc_list)))
    print("Edit Score: {:.2f}%".format(100 * np.mean(edit_list)))

    for i, threshold in enumerate(f1_thresholds):
        scores = [f1_scores[j][i] for j in range(len(f1_scores))]
        print(f"F1@{threshold:.2f}: {100 * np.mean(scores):.2f}%")
    
    # --- --- --- ---

if __name__ == "__main__":
    evaluate_classifier_segmentation()
