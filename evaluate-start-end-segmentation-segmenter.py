# HYDRA_FULL_ERROR=1 python evaluate-start-end-segmentation-segmenter.py --multirun \
#     data.dir=/home/nadir/windowed-babel-for-classification-for-validation \
#     window_step=1 \
#     vote_manager=score-based \
#     run_dir=outputs/start-end-segmentation_None_tmr_majority-based-start-end-with-majority_False_mlp_20
    
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

F1_THRESHOLDS = np.arange(0.1, 1.1, 0.1)

@hydra.main(version_base=None, config_path="configs", config_name="evaluate-start-end-segmentation-segmenter")
def evaluate_start_end_segmentation_segmenter(newcfg: DictConfig) -> None:
    device = newcfg.device
    run_dir = newcfg.run_dir
    examples = newcfg.examples
    ckpt_name = newcfg.ckpt

    save_dir = os.path.join(run_dir, "segmentation-evaluation")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[newcfg.data]: {newcfg.data}")
    
    window_size = int(run_dir.split("_mlp_")[1])
    window_step = int(newcfg.window_step) if f"{newcfg.window_step}".isdigit() else window_size if newcfg.window_step == "window_size" else 1
    
    print(f"[window_size]: {window_size}")
    print(f"[window_step]: {window_step}")
    
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

    all_predictions, all_groundtruths = [], []
    
    with torch.no_grad():
        for index, sample in tqdm.tqdm(iterable=enumerate(dataset), total=len(dataset), desc="[evaluate-segmentation]"):
            sample["motion"] = sample["motion"].to(device)
            sample["transformed_motion"] = sample["transformed_motion"].to(device)
            sample["annotation"] = sample["annotation"].to(device)
            
            label = sample["annotation"]
            
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
            
            prediction = outputs.cpu().numpy()
            groundtruth = label.cpu().numpy()
            
            all_predictions.append(prediction)
            all_groundtruths.append(groundtruth)
            
    accuracies_list = []
    editscores_list = []
    f1_scores = []    
    
    from src.model.metrics import accuracy_score, levenshtein, f_score

    for predicted_sequence, groundtruth_sequence in zip(all_predictions, all_groundtruths):
        accuracy = accuracy_score(groundtruth_sequence, predicted_sequence)
        # editscore = levenshtein(predicted_sequence, groundtruth_sequence)
        editscore = levenshtein(groundtruth_sequence, predicted_sequence)
        # f1_thresholds_scores = list(map(lambda threshold: f_score(predicted_sequence, groundtruth_sequence, overlap=threshold), F1_THRESHOLDS))
        f1_thresholds_scores = list(map(lambda threshold: f_score(groundtruth_sequence, predicted_sequence, overlap=threshold), F1_THRESHOLDS))
        
        accuracies_list.append(accuracy)
        editscores_list.append(editscore)
        f1_scores.append(f1_thresholds_scores)

    print("Frame-wise Accuracy: {:.2f}%".format(100 * np.mean(accuracies_list)))
    print("Edit Score: {:.2f}%".format(100 * np.mean(editscores_list)))

    for i, threshold in enumerate(F1_THRESHOLDS):
        scores = [f1_scores[j][i] for j in range(len(f1_scores))]
        print(f"F1@{threshold:.2f}: {100 * np.mean(scores):.2f}%")
        
    average_f1_score = np.mean(f1_scores)
    
    print(f"Average F1 Score: {100 * average_f1_score:.2F}")
        
    evaluation_metrics: dict = {
        "framewise_accuracy": float(np.mean(accuracies_list)),
        "edit_score": float(np.mean(editscores_list)),
        "average_f1_score": average_f1_score,
        "f1_scores": {
            f"f1@{threshold:.2f}": float(np.mean([f1_scores[j][i] for j in range(len(f1_scores))]))
            for i, threshold in enumerate(F1_THRESHOLDS)
        }
    }
    
    # --- --- ---
    
    flat_predictions = np.concatenate([prediction.flatten() for prediction in all_predictions])
    flat_groundtruths = np.concatenate([groundtruth.flatten() for groundtruth in all_groundtruths])

    unique_classes = np.unique(np.concatenate([flat_predictions, flat_groundtruths]))

    per_class_falsepositive_rates = {}
    total_frames = len(flat_predictions)
    
    for cls in unique_classes:
        # NOTE: predicted this class, but ground truth is not this class
        falsepositive_frames = np.logical_and(flat_predictions == cls, flat_groundtruths != cls).sum()
        falsepositive_rate = falsepositive_frames / total_frames if total_frames > 0 else 0.0
        
        per_class_falsepositive_rates[f"class_{int(cls)}"] = float(falsepositive_rate)
        
        print(f"\tClass {int(cls)} False Positive Rate: {100 * falsepositive_rate:.2f}%")

    overall_fp_rate = np.mean(list(per_class_falsepositive_rates.values()))
    
    print(f"Overall False Positive Rate: {100 * overall_fp_rate:.2f}%")

    evaluation_metrics["per_class_false_positive_rates"] = per_class_falsepositive_rates
    evaluation_metrics["overall_false_positive_rate"] = float(overall_fp_rate)
    
    # --- --- ---
    
    segments_count_accuracies = []

    for groundtruth_sequence, predicted_sequence in zip(all_groundtruths, all_predictions):
        groundtruth_segments_count = len(extract_segments(groundtruth_sequence))
        prediction_segments_count = len(extract_segments(predicted_sequence))
        
        if groundtruth_segments_count > 0:
            segments_count_accuracy = 1.0 - abs(prediction_segments_count - groundtruth_segments_count) / groundtruth_segments_count
        else:
            segments_count_accuracy = 1.0

        segments_count_accuracies.append(segments_count_accuracy)

    mean_segments_count_accuracy = np.mean(segments_count_accuracies)

    print(f"Segments Count Accuracy: {100 * mean_segments_count_accuracy:.2f}%")

    evaluation_metrics["segments_count_accuracy"] = float(mean_segments_count_accuracy)
        
    # --- --- ---

    metrics_path = os.path.join(save_dir, f"metrics-{window_size}-{window_step}-{newcfg['vote_manager']['_target_'].split('.')[-1]}")
    
    save_metric(metrics_path, evaluation_metrics, format="yaml")
    save_metric(metrics_path, evaluation_metrics, format="json")
    
    print(f"\n[metrics-saved-to]: {metrics_path}")
        
if __name__ == "__main__":
    evaluate_start_end_segmentation_segmenter()
