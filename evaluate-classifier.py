# python evaluate-classifier.py run_dir=/home/nadir/disk/codes/tmr-code/outputs/classifier_babel-classifier_guoh3dfeats

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

@hydra.main(version_base=None, config_path="configs", config_name="evaluate-classifier")
def evaluate_classifier(newcfg: DictConfig) -> None:
    device = newcfg.device
    run_dir = newcfg.run_dir
    examples = newcfg.examples
    ckpt_name = newcfg.ckpt

    save_dir = os.path.join(run_dir, "classification-evaluation")
    os.makedirs(save_dir, exist_ok=True)

    cfg = read_config(run_dir)

    pl.seed_everything(cfg.seed)

    logger.info("[model]: loading")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    dataset = instantiate(cfg.data, split="test")
    
    dataloader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    model.eval()

    # --- --- --- ---
    
    all_predictions, all_labels = [], []
    
    from src.model import ClassifierModel

    with torch.no_grad():
        for index, batch in tqdm.tqdm(iterable=enumerate(dataloader), total=len(dataloader), desc="[evaluate-segmentation]"):
            batch["motion_x_dict"]["x"] = batch["motion_x_dict"]["x"].to(device)
            batch["motion_x_dict"]["mask"] = batch["motion_x_dict"]["mask"].to(device)
            
            outputs = model(batch, None)
            
            labels = ClassifierModel.get_targets(batch, None)
            predictions = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
            
            for prediction, label in zip(predictions, labels):
                all_predictions.append(prediction)
                all_labels.append(label)

    from src.model.metrics import accuracy_score, precision_score, recall_score, f1_score

    # TODO: compute balanced accuracy score
    accuracy = accuracy_score(all_predictions, all_labels)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"[accuracy]: {(accuracy * 100):.02f}%")
    print(f"[precision]: {(precision * 100):.02f}%")
    print(f"[recall]: {(recall * 100):.02f}%")
    print(f"[f1]: {(f1 * 100):.02f}%")
    
    from sklearn.metrics import confusion_matrix as generate_confusion_matrix

    confusion_matrix = generate_confusion_matrix(all_labels, all_predictions)
    
    print("--- --- ---")
    
    print(confusion_matrix)
    
    # --- --- --- ---

if __name__ == "__main__":
    evaluate_classifier()
