# HYDRA_FULL_ERROR=1 python visualize-classification-result.py \
#     max_epochs=null \
#     run_dirs=[/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_10,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_15,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_20,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_25,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_majority-based-start-end-with-majority_False_mlp_30,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_10,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_15,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_20,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_25,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-with-majority_False_mlp_30,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_10,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_15,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_20,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_25,/home/nadir/tmr-code/outputs/start-end-segmentation_tmr_transition-based-start-end-without-majority_False_mlp_30]

import os
import re
import yaml
import json
import hydra
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from omegaconf import DictConfig

def load_metrics(output_path):
    metrics_path_json = os.path.join(output_path, "classification-evaluation", "metrics.json")
    metrics_path_yaml = os.path.join(output_path, "classification-evaluation", "metrics.yaml")
    
    try:
        if os.path.exists(metrics_path_json):
            with open(metrics_path_json, "r") as f:
                return json.load(f)
        elif os.path.exists(metrics_path_yaml):
            with open(metrics_path_yaml, "r") as f:
                return yaml.safe_load(f)
    except Exception as exception:
        print(f"[warning]: error loading metrics from {output_path}: {exception}")
        
    return None

def get_model_key(config: dict) -> str:
    encoder_type = config["model"]["motion_encoder"]["_target_"].split(".")[-1]
    label_extractor = config["model"]["label_extractor"]["_target_"].split(".")[-1]
    pretrained = config["model"]["motion_encoder"].get("pretrained", False)
    classifier = config["model"]["classifier"]["_target_"].split(".")[-1]
    window_size = config["data"]["window_size"]
    
    return f"{encoder_type} with {label_extractor} ({pretrained} - {window_size})"

def get_epoch_loss(epoch_rows: pd.DataFrame, loss_epoch_col: str, loss_step_col: str) -> float:
    if loss_step_col in epoch_rows.columns:
        loss_step = epoch_rows[loss_step_col].dropna()
        if not loss_step.empty:
            return loss_step.iloc[-1]
    
    if loss_epoch_col in epoch_rows.columns:
        loss_epoch = epoch_rows[loss_epoch_col].dropna()
        if not loss_epoch.empty:
            return loss_epoch.iloc[-1]
            
    return np.nan

def process_run_data(output_path: str, max_epochs: int | None = None):
    try:
        config_path = os.path.join(output_path, "config.json")
        log_path = os.path.join(output_path, "logs", "metrics.csv")
        
        if not all(os.path.exists(p) for p in [config_path, log_path]):
            print(f"[warning]: skipping {output_path} due to missing config.json or metrics.csv")
            return None

        with open(config_path, 'r') as file:
            config = json.load(file)
            
        log = pd.read_csv(log_path)

        run_info = {
            "path": output_path,
            "model_key": get_model_key(config),
            "window_size": config["data"]["window_size"]
        }
        
        epoch_data = log[log['epoch'].notna()].copy()
        epochs = sorted(epoch_data['epoch'].unique())
        
        print("[len(epochs)]:", len(epochs))
        
        if max_epochs is not None:
            epochs = [epoch for epoch in epochs if epoch <= max_epochs]
        
        if not epochs:
            print(f"[warning]: no epochs to plot for {output_path}")
            return None
        
        run_info["epochs"] = epochs
        
        losses = collections.defaultdict(list)
        loss_types = [
            'train_loss',
            'val_loss',
            'train_class_loss',
            'train_start_loss', 
            'train_end_loss',
            'val_class_loss',
            'val_start_loss',
            'val_end_loss'
        ]

        for epoch in epochs:
            epoch_rows = epoch_data[epoch_data['epoch'] == epoch]
            for loss_type in loss_types:
                loss_val = get_epoch_loss(epoch_rows, f'{loss_type}_epoch', f'{loss_type}_step')
                losses[loss_type].append(loss_val)
        
        run_info.update(losses)
        return run_info

    except Exception as e:
        print(f"error processing data for {output_path}: {e}")
        return None

@hydra.main(version_base=None, config_path="configs", config_name="visualize-classification-results")
def main(cfg: DictConfig):
    max_epochs = cfg.max_epochs
    output_paths = cfg.run_dirs
    
    all_run_data = [data for path in output_paths if (data := process_run_data(path, max_epochs)) is not None]
    
    if not all_run_data:
        print("no valid run data found to plot. Exiting.")
        return

    model_labels = sorted(list(set(d['model_key'] for d in all_run_data)))
    window_sizes = sorted(list(set(d['window_size'] for d in all_run_data)))
    
    rows = len(model_labels) if model_labels else 1
    cols = len(window_sizes) if window_sizes else 1
    
    data_grid = {(d['model_key'], d['window_size']): d for d in all_run_data}

    # NOTE: overall training and validation loss
    fig_overall, axes_overall = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5), squeeze=False, constrained_layout=True)
    fig_overall.suptitle("Overall Loss Analysis", fontsize=20, weight='bold')

    all_losses = []
    for d in all_run_data:
        for key in ['train_loss', 'val_loss']:
            if d.get(key):
                all_losses.extend(d[key])
    
    valid_losses = [loss for loss in all_losses if pd.notna(loss)]
    global_min = np.min(valid_losses) if valid_losses else 0
    global_max = np.max(valid_losses) if valid_losses else 1

    for r, model_label in enumerate(model_labels):
        for c, window_size in enumerate(window_sizes):
            ax = axes_overall[r, c]
            run_data = data_grid.get((model_label, window_size))

            if run_data is None:
                ax.axis('off')
                continue

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

            if c == 0: ax.set_ylabel(model_label, fontsize=14, weight='bold')
            if r == 0: ax.set_title(f"Window Size = {window_size}", fontsize=14, weight='bold')

            epochs = run_data['epochs']
            handles, labels = [], []

            train_losses = run_data.get('train_loss', [])
            if any(pd.notna(l) for l in train_losses):
                p_train, = ax.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
                handles.append(p_train)
                labels.append('Training Loss')

            val_losses = run_data.get('val_loss', [])
            if any(pd.notna(l) for l in val_losses):
                p_val, = ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
                handles.append(p_val)
                labels.append('Validation Loss')

                val_losses_np = np.array(val_losses, dtype=float)
                if np.any(~np.isnan(val_losses_np)):
                    min_val_idx = np.nanargmin(val_losses_np)
                    min_val_epoch = epochs[min_val_idx]
                    min_val_loss = val_losses_np[min_val_idx]
                    ax.axhline(min_val_loss, color='red', linestyle='--', linewidth=1, alpha=0.7)
                    ax.axvline(min_val_epoch, color='red', linestyle='--', linewidth=1, alpha=0.7)
                    ax.annotate(f"Min: {min_val_loss:.3f} @ E{int(min_val_epoch)}",
                                xy=(min_val_epoch, min_val_loss), xytext=(5, -15),
                                textcoords='offset points', fontsize=8, color='red')

            ax.grid(True, alpha=0.3)
            ax.set_ylim(global_min * 0.95, global_max * 1.05)
            if handles: ax.legend(handles, labels, fontsize=8)
            
            metrics = load_metrics(run_data['path'])
            acc_text = "No metrics found"
            if metrics:
                acc = metrics.get("overall_accuracy", 0) * 100
                acc_text = f"Accuracy: {acc:.2f}%"
            ax.text(0.05, 0.95, acc_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # NOTE: detailed component losses
    fig_detailed, axes_detailed = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5), squeeze=False, constrained_layout=True)
    fig_detailed.suptitle("Detailed Validation Loss Components", fontsize=20, weight='bold')

    for r, model_label in enumerate(model_labels):
        for c, window_size in enumerate(window_sizes):
            ax = axes_detailed[r, c]
            run_data = data_grid.get((model_label, window_size))

            if run_data is None:
                ax.axis('off')
                continue
                
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

            if c == 0: ax.set_ylabel(model_label, fontsize=14, weight='bold')
            if r == 0: ax.set_title(f"Window Size = {window_size}", fontsize=14, weight='bold')

            epochs = run_data['epochs']
            handles, labels = [], []

            loss_components = {
                'val_class_loss': ('b-s', 'Val Class Loss'),
                'val_start_loss': ('g-s', 'Val Start Loss'),
                'val_end_loss':   ('r-s', 'Val End Loss'),
            }
            for key, (style, label) in loss_components.items():
                losses = run_data.get(key, [])
                if any(pd.notna(l) for l in losses):
                    p, = ax.plot(epochs, losses, style, label=label, markersize=4)
                    handles.append(p)
                    labels.append(label)
            
            ax.grid(True, alpha=0.3)
            if handles: ax.legend(handles, labels, fontsize=9)

    save_dir = "outputs/training_results_visualization_classification"
    os.makedirs(save_dir, exist_ok=True)
    
    overall_path = os.path.join(save_dir, 'overall_loss_grid.png')
    detailed_path = os.path.join(save_dir, 'individual_loss_grid.png')
    
    fig_overall.savefig(overall_path, dpi=300, bbox_inches='tight')
    fig_detailed.savefig(detailed_path, dpi=300, bbox_inches='tight')
    
    print(f"[info]: saved overall loss visualization to: {overall_path}")
    print(f"[info]: saved detailed loss visualization to: {detailed_path}")
    
    plt.show()

if __name__ == "__main__":
    main()