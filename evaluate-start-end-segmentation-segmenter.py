# HYDRA_FULL_ERROR=1 python evaluate-start-end-segmentation-segmenter.py --multirun \
#     data.dir=/home/nadir/disk/datasets/babel-for-validation \
#     window_step=1,window_size \
#     vote_manager=class-based,score-based \
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
from src.logging import save_metric
from hydra.utils import instantiate
from src.load import load_model_from_cfg

logger = logging.getLogger(__name__)

def extract_segments(sequence):
    """
    Convert a label sequence to a list of segments.
    Each segment is a tuple: (label, start_idx, end_idx).
    """
    segments = []
    prev_label = sequence[0]
    start_idx = 0
    for i in range(1, len(sequence)):
        if sequence[i] != prev_label:
            segments.append((prev_label, start_idx, i))
            start_idx = i
            prev_label = sequence[i]
    segments.append((prev_label, start_idx, len(sequence)))
    return segments

@hydra.main(version_base=None, config_path="configs", config_name="evaluate-start-end-segmentation-segmenter")
def evaluate_start_end_segmentation_segmenter(newcfg: DictConfig) -> None:
    device = newcfg.device
    run_dir = newcfg.run_dir
    examples = newcfg.examples
    ckpt_name = newcfg.ckpt

    save_dir = os.path.join(run_dir, "segmentation-evaluation")
    os.makedirs(save_dir, exist_ok=True)
    
    print(newcfg.data)
    
    window_size = int(run_dir.split("_mlp_")[1])
    window_step = int(newcfg.window_step) if f"{newcfg.window_step}".isdigit() else window_size if newcfg.window_step == "window_size" else 1
    
    print("[window_size]:", window_size)
    print("[window_step]:", window_step)
    
    vote_manager = instantiate(newcfg.vote_manager)
    
    dataset = instantiate(newcfg.data, window_size=window_size, split="all", for_validation=True)
    
    normalization_statistics = torch.load(os.path.join(dataset.dir, "motion_normalization_stats.pt"))

    mean = normalization_statistics["mean"]
    std = normalization_statistics["std"]
    
    print("[mean.shape]:", mean.shape)
    print("[std.shape]:", std.shape)

    # NOTE: will load the config used to train the model
    cfg = read_config(run_dir)

    pl.seed_everything(cfg.seed)

    logger.info("[model]: loading")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    model = model.eval()

    all_preds, all_labels = [], []
    
    # print("[#dataset]:", len(dataset))
    
    # key = "annotation"
    key = "transition_mask"

    with torch.no_grad():
        for index, sample in tqdm.tqdm(iterable=enumerate(dataset), total=len(dataset), desc="[evaluate-segmentation]"):
            sample["transformed_motion"] = sample["transformed_motion"].to(device)
            
            sample["motion"] = sample["motion"].to(device)
            sample[key] = sample[key].to(device)
            
            sample["annotation"] = sample[key]
            
            # x = sample["motion_x_dict"]
            y = sample[key]
            
            outputs, exception = model.segment_sequence(
                sample,
                window_size=window_size,
                window_step=window_step,
                vote_manager=vote_manager,
                mean=mean,
                std=std
            )
            
            if exception is not None:
                logger.warning(f"[skipped-sequence]: {index} due to {exception}")
                continue
            
            # print("[outputs.shape]:", outputs.shape)
            
            preds = outputs.cpu().numpy()
            labels = y.cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)
            
            # labels = y.squeeze(1).cpu().numpy()
            # for pred_seq, label_seq in zip(preds, labels):
            #     all_preds.append(pred_seq)
            #     all_labels.append(label_seq)

    acc_list = []
    edit_list = []
    
    f1_thresholds = np.arange(0.1, 1.1, 0.1)
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
        
    evaluation_metrics: dict = {
        "framewise_accuracy": float(np.mean(acc_list)),
        "edit_score": float(np.mean(edit_list)),
        "f1_scores": {
            f"f1@{threshold:.2f}": float(np.mean([f1_scores[j][i] for j in range(len(f1_scores))]))
            for i, threshold in enumerate(f1_thresholds)
        }
    }
    
    # --- --- ---
    
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_labels = np.concatenate([l.flatten() for l in all_labels])

    # NOTE: False Positive: predicted 1, ground truth is 0
    fp_frames = np.logical_and(flat_preds == 1, flat_labels == 0).sum()
    total_frames = len(flat_preds)

    frame_fp_rate = fp_frames / total_frames if total_frames > 0 else 0.0

    print(f"Frame-level False Positive Rate: {100 * frame_fp_rate:.2f}%")

    evaluation_metrics["false_positive_score"] = float(frame_fp_rate)
    
    # --- --- ---
    
    transition_accuracies = []

    def count_transitions(seq):
        return len(extract_segments(seq)) - 1

    for pred_seq, label_seq in zip(all_preds, all_labels):
        gt_transitions = count_transitions(label_seq)
        pred_transitions = count_transitions(pred_seq)
        
        if gt_transitions > 0:
            acc = 1.0 - abs(pred_transitions - gt_transitions) / gt_transitions
        else:
            acc = 1.0 if pred_transitions == 0 else 0.0

        transition_accuracies.append(acc)

    mean_transition_acc = np.mean(transition_accuracies)

    print(f"Transition Count Accuracy: {100 * mean_transition_acc:.2f}%")

    evaluation_metrics["transition_count_accuracy"] = float(mean_transition_acc)
    
    # --- --- ---
    
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        balanced_accuracy_score,
        matthews_corrcoef
    )
    
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_labels = np.concatenate([l.flatten() for l in all_labels])
    
    report = classification_report(flat_labels, flat_preds, output_dict=True, zero_division=0)
    balanced_acc = balanced_accuracy_score(flat_labels, flat_preds)
    mcc = matthews_corrcoef(flat_labels, flat_preds)

    # Optionally, also compute and print confusion matrix
    conf_matrix = confusion_matrix(flat_labels, flat_preds, normalize='true')
    
    print("\n--- Additional Evaluation Metrics (for imbalanced data) ---")
    print(f"Balanced Accuracy: {100 * balanced_acc:.2f}%")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print("Per-class Precision, Recall, F1:")
    for cls in report:
        if cls not in ("accuracy", "macro avg", "weighted avg"):
            print(f"Class {cls}: "
                f"Precision={100 * report[cls]['precision']:.2f}%, "
                f"Recall={100 * report[cls]['recall']:.2f}%, "
                f"F1={100 * report[cls]['f1-score']:.2f}%")
            
    evaluation_metrics.update({
        "balanced_accuracy": float(balanced_acc),
        "matthews_corrcoef": float(mcc),
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"])
            }
            for cls in report
            if cls not in ("accuracy", "macro avg", "weighted avg")
        }
    })
    
    # --- --- ---

    metrics_path = os.path.join(save_dir, f"metrics-{window_size}-{window_step}-{newcfg['vote_manager']['_target_'].split('.')[-1]}")
    
    save_metric(metrics_path, evaluation_metrics, format="yaml")
    save_metric(metrics_path, evaluation_metrics, format="json")
    
    print(f"\n[metrics-saved-to]: {metrics_path}")
        
if __name__ == "__main__":
    evaluate_start_end_segmentation_segmenter()
