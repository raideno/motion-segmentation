# HYDRA_FULL_ERROR=1 python visualize-segmentation-results.py --multirun \
#     +data.dir=/home/nadir/disk/datasets/babel-for-validation \
#     window_step=1,window_size \
#     +vote_manager=score-based,class-based \
#     qualitative_run_index=2 \
#     num_qualitative_examples=32 \
#     run_dirs=[/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_10,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_15,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_20,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_25,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_30,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_10,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_15,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_20,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_25,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_30,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_10,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_15,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_20,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_25,/home/nadir/tmr-code/outputs/archives.new/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_30]

# HYDRA_FULL_ERROR=1 python visualize-segmentation-results.py --multirun \
#     +data.dir=/home/nadir/windowed-babel-for-classification-for-validation \
#     window_step=1 \
#     +vote_manager=score-based \
#     qualitative_run_index=0 \
#     num_qualitative_examples=128 \
#     run_dirs=[/home/nadir/tmr-code/outputs/multi-class-model-super-super-06-17-10:22]

import os
import yaml
import json
import torch
import hydra
import logging
import colorlog
import collections
import logging.config

import src.prepare

import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from src.config import read_config
from hydra.utils import instantiate
from hydra import compose, initialize
from src.load import load_model_from_cfg
from hydra.core.hydra_config import HydraConfig
from tas_helpers.visualization import SegmentationVisualizer
from tas_helpers.utils import frame_level_annotations_to_segment_level_annotations

logger = logging.getLogger(__name__)

def load_segmentation_metrics(
    output_path,
    window_size,
    window_step,
    vote_manager
):
    """
    Load segmentation evaluation metrics from JSON file.
    """
    metrics_path = os.path.join(output_path, "segmentation-evaluation", f"metrics-{window_size}-{window_step}-{vote_manager}.json")
    
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading metrics from {output_path}: {e}")
    return None

# def parse_model_info_from_path(output_path: str):
#     # start-end-segmentation/////tmr/////majority-based-start-end-with-majority/////False/////mlp/////10
#     # _, encoder_type, label_extractor, pretrained, classifier, window_size = os.path.basename(output_path).split('_')
#     _, encoder_type, _, label_extractor, pretrained, classifier, window_size = os.path.basename(output_path).split('_')
    
#     return {
#         "name": f"{encoder_type} with {label_extractor}",
#         "window_size": int(window_size),
#         "classifier": classifier,
#         "pretrained": pretrained,
#         "label_extractor": label_extractor,
#         "encoder_type": encoder_type,
#     }

def parse_model_info_from_path(output_path: str):
    config_path = os.path.join(output_path, "config.json")
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    encoder_type = config["model"]["motion_encoder"]["_target_"].split(".")[-1]
    label_extractor = config["model"]["label_extractor"]["_target_"].split(".")[-1]
    pretrained = config["model"]["motion_encoder"].get("pretrained", False)
    classifier = config["model"]["classifier"]["_target_"].split(".")[-1]
    window_positional_encoder = config["model"]["window_positional_encoder"]["_target_"].split(".")[-1]
    
    window_size = config["data"]["window_size"]
    
    name = f"{encoder_type} with {label_extractor}",
    
    return {
        "name": f"{encoder_type} with {label_extractor}",
        "window_size": int(window_size),
        "classifier": classifier,
        "pretrained": pretrained,
        "label_extractor": label_extractor,
        "encoder_type": encoder_type,
    }

def prepare_model_and_data(cfg: DictConfig, run_dir: str):
    device = cfg.device
    ckpt_name = cfg.ckpt

    training_configuration = read_config(run_dir)
    pl.seed_everything(training_configuration.seed)

    model = load_model_from_cfg(training_configuration, ckpt_name, eval_mode=True, device=device, pretrained=True)
    model = model.eval()
    
    window_size = training_configuration.data.window_size
    
    dataset = instantiate(
        {
            "_target_": "src.data.windowed_dataset.WindowedDataset",
        },
        # dir="/home/nadir/disk/datasets/babel-for-validation/",
        dir="/home/nadir/windowed-babel-for-classification-for-validation",
        window_size=window_size,
        split="all",
        for_validation=True,
        normalize=False,
        balanced=False
    )
    
    normalization_statistics = torch.load(os.path.join(dataset.dir, "motion_normalization_stats.pt"))
    
    mean = normalization_statistics["mean"]
    std = normalization_statistics["std"]
    
    return model, dataset, device, window_size, mean, std

def run_qualitative_analysis(cfg: DictConfig, save_dir: str):
    try:
        run_dir_to_visualize = cfg.run_dirs[cfg.qualitative_run_index]
    except IndexError:
        logger.error(f"qualitative_run_index {cfg.qualitative_run_index} is out of bounds.")
        return

    model, dataset, device, window_size, mean, std = prepare_model_and_data(cfg, run_dir_to_visualize)

    random_indices = np.random.choice(len(dataset), size=cfg.num_qualitative_examples, replace=False)
    visualizer = SegmentationVisualizer(labels_values=range(0,23))
    vote_manager = instantiate(cfg.vote_manager)

    for seq_index in random_indices:
        sample = dataset[seq_index]
        sample["transformed_motion"] = sample["transformed_motion"].to(device)
        sample["motion"] = sample["motion"].to(device)
        sample["annotation"] = sample["annotation"].to(device)
        
        outputs, exception = model.segment_sequence(
            sample,
            vote_manager=vote_manager,
            window_size=window_size,
            window_step=1,
            mean=mean,
            std=std
        )

        if exception is not None:
            logger.warning(f"Skipped sequence {seq_index} due to: {exception}")
            continue

        preds = outputs.cpu().numpy()
        labels = sample["annotation"].cpu().numpy()
        
        labels = sample["annotation"].cpu().numpy()
        labels[labels < 0] = -labels[labels < 0] + 20
        
        # --- --- ---
        
        temp_labels_path = os.path.join(save_dir, f"temp_labels_{seq_index}.png")
        visualizer.plot_segmentation(labels, header="Ground Truth", fps=20, show_ticks=True, show_legend=False)
        plt.savefig(temp_labels_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Generate and save the prediction plot to a temporary file
        temp_preds_path = os.path.join(save_dir, f"temp_preds_{seq_index}.png")
        visualizer.plot_segmentation(preds, header="Prediction", fps=20, show_ticks=True, show_legend=False)
        plt.savefig(temp_preds_path, dpi=150, bbox_inches='tight')
        plt.close()

        img_labels = plt.imread(temp_labels_path)
        img_preds = plt.imread(temp_preds_path)

        final_fig, final_axes = plt.subplots(2, 1, figsize=(15, 6))
        final_fig.suptitle(f"Segmentation Example (Sequence: {seq_index}, Model: {os.path.basename(run_dir_to_visualize)})", fontsize=16)

        final_axes[0].imshow(img_labels)
        final_axes[0].axis('off')

        final_axes[1].imshow(img_preds)
        final_axes[1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        final_save_path = os.path.join(save_dir, f"example_seq_{seq_index}.png")
        plt.savefig(final_save_path, dpi=150)
        plt.close(final_fig)
        
        logger.info(f"saved combined visualization to {final_save_path}")

        os.remove(temp_labels_path)
        os.remove(temp_preds_path)

        # --- --- ---

def run_quantitative_analysis(cfg: DictConfig, save_dir: str):
    all_metrics = []
    model_data = collections.defaultdict(lambda: collections.defaultdict(dict))

    for output_path in cfg.run_dirs:
        model_info = parse_model_info_from_path(output_path)
        
        model_name = model_info['name']
        window_size = model_info['window_size']
        
        window_step = int(cfg.window_step) if f"{cfg.window_step}".isdigit() else window_size if cfg.window_step == "window_size" else 1
        vote_manager = cfg.vote_manager["_target_"].split('.')[-1]
        
        metrics = load_segmentation_metrics(
            output_path=output_path,
            window_size=window_size,
            window_step=window_step,
            vote_manager=vote_manager
        )
        if metrics:
            all_metrics.append({
                'model': model_name, 'window_size': window_size,
                **metrics, **metrics['f1_scores'],
            })
            model_data[model_name][window_size] = metrics

    if not all_metrics:
        logger.warning("no metrics found. Skipping quantitative analysis.")
        return

    df = pd.DataFrame(all_metrics)
    # df = df.drop(columns=['f1_scores', 'per_class'])
    
    logger.info(f"loaded metrics for {len(all_metrics)} configurations.")
    logger.info(f"models: {df['model'].unique()}")
    logger.info(f"window sizes: {sorted(df['window_size'].unique())}")
    
    sns.set_palette("husl")
    f1_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    
    # NOTE: Key Metrics vs. Window Size
    key_metrics_to_plot = [
        ('segments_count_accuracy', 'Segments Count Accuracy'),
        ('framewise_accuracy', 'Framewise Accuracy'),
        ('edit_score', 'Edit Score'),
        ('average_f1_score', 'Average F1 Score'),
        ('overall_false_positive_rate', 'Overall False Positive Rate')
    ]
    fig, axes = plt.subplots(1, len(key_metrics_to_plot), figsize=(24, 5), sharex=True)
    fig.suptitle('Model Performance Metrics vs Window Size', fontsize=18, fontweight='bold')
    for ax, (metric_key, metric_label) in zip(axes, key_metrics_to_plot):
        sns.lineplot(data=df, x='window_size', y=metric_key, hue='model', marker='o', ax=ax)
        ax.set_title(metric_label, fontweight='bold')
        ax.set_xlabel('Window Size')
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "key_metrics_vs_window_size.png"), dpi=300)
    plt.close(fig)
    
    # NOTE: F1 Score at each threshold vs. Window Size
    num_thresholds = len(f1_thresholds)
    ncols = min(5, num_thresholds)
    nrows = int(np.ceil(num_thresholds / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    fig.suptitle('Window Size vs F1 Score for Each IoU Threshold', fontsize=18, fontweight='bold')
    axes = axes.flatten()
    for idx, threshold in enumerate(f1_thresholds):
        ax = axes[idx]
        threshold_col = f'f1@{threshold:.2f}'
        sns.lineplot(data=df, x='window_size', y=threshold_col, hue='model', marker='o', ax=ax)
        ax.set_title(f'IoU ≥ {threshold:.2f}', fontweight='bold')
        ax.set_xlabel('Window Size')
        ax.set_ylabel('F1 Score')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')
    for i in range(len(f1_thresholds), len(axes)): fig.delaxes(axes[i])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "f1_score_by_threshold_vs_window_size.png"), dpi=300)
    plt.close(fig)
    
    # NOTE: F1 Score vs. IoU Threshold (one subplot per model)
    models = df['model'].unique()
    window_sizes = sorted(df['window_size'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(20, 6), sharey=True)
    if len(models) == 1: axes = [axes] # Ensure axes is iterable
    fig.suptitle('F1 Scores at Different IoU Thresholds', fontsize=16, fontweight='bold')
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        for window_size in window_sizes:
            if model in model_data and window_size in model_data[model]:
                metrics = model_data[model][window_size]
                f1_values = [metrics['f1_scores'][f'f1@{thresh:.2f}'] for thresh in f1_thresholds]
                ax.plot(f1_thresholds, f1_values, marker='o', label=f'Window {window_size}')
        ax.set_title(f'{model}', fontweight='bold'); ax.set_xlabel('IoU Threshold'); ax.set_ylabel('F1 Score')
        ax.grid(True, alpha=0.3); ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "f1_score_vs_iou_threshold.png"), dpi=300)
    plt.close(fig)

    logger.info("All quantitative plots saved.")

    # NOTE: print summaries
    print("\n" + "="*80 + "\nSUMMARY STATISTICS\n" + "="*80)
    for model in models:
        print(f"\n{model}:")
        model_df = df[df['model'] == model]
        for metric in ['transition_count_accuracy', 'framewise_accuracy', 'edit_score', 'balanced_accuracy', 'f1@0.50', 'f1@1.00']:
            if metric in model_df:
                best_idx = model_df[metric].idxmax()
                print(f"  Best {metric.replace('_', ' ').title()}: {model_df.loc[best_idx, metric]:.4f} (Window {model_df.loc[best_idx, 'window_size']})({best_idx})")
    
    print(f"\n{'='*80}\nBEST OVERALL CONFIGURATIONS\n{'='*80}")
    for metric in ['transition_count_accuracy', 'framewise_accuracy', 'edit_score', 'f1@0.50', 'f1@0.90', 'f1@1.00', 'balanced_accuracy']:
        if metric in df:
            best_idx = df[metric].idxmax()
            best_config = df.loc[best_idx]
            print(f"\nBest {metric.replace('_', ' ').title()}: {best_config[metric]:.4f}")
            print(f"  Model: {best_config['model']}({best_idx})\n  Window Size: {best_config['window_size']}")

@hydra.main(version_base=None, config_path="configs", config_name="visualize-segmentation-results")
def main(cfg: DictConfig):
    output_dir = "outputs/training_results_visualization_segmentation"
    
    window_step = cfg.window_step
    vote_manager = cfg.vote_manager["_target_"].split('.')[-1]
    
    examples_dir = os.path.join(output_dir, "examples")
    quantitative_dir = os.path.join(output_dir, f"quantitative-{window_step}-{vote_manager}")
    
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(quantitative_dir, exist_ok=True)
    
    logger.info(f"starting segmentation analysis. Outputs will be saved to '{output_dir}'")
    
    run_quantitative_analysis(cfg, save_dir=quantitative_dir)
    
    run_qualitative_analysis(cfg, save_dir=examples_dir)

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()