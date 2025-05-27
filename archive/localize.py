import os
import hydra
import logging

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def load_models(cfg: DictConfig, ckpt_name, device):
    from hydra.utils import instantiate
    from src.load import load_model_from_cfg
    
    logger.info("Loading the token-embedding model")
    token_embedding_model = instantiate(cfg.data.text_to_token_emb, device=device)

    logger.info("Loading the text-embedding model")
    text_embedding_model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    
    logger.info("Loading the motion-normalization model")
    motion_normalization_model = instantiate(cfg.data.motion_loader.normalizer)

    logger.info("Loading the motion-embedding model")
    motion_embedding_model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    
    return token_embedding_model, text_embedding_model, motion_normalization_model, motion_embedding_model

def compute_similarity(text_embedding, motion_embedding):
    dot_product = np.dot(text_embedding, motion_embedding)
    
    text_norm = np.linalg.norm(text_embedding)
    motion_norm = np.linalg.norm(motion_embedding)
    
    similarity = dot_product / (text_norm * motion_norm)
    
    return similarity

def visualize_results(file, text, similarity_scores):
    # print(f"[motion_length]: {motion_length}")
    # print(f"[window_size]: {window_size}")
    # print(f"[text_embedding.shape]: {text_embedding.shape}")
    # print(f"[window_motion_embedding.shape]: {motion_window_embedding.shape}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(similarity_scores)
    plt.title(f'Text-Motion Similarity Over Time\nText: "{text}"')
    plt.xlabel('Frame Index')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    
    max_sim_idx = np.argmax(similarity_scores)
    max_sim_value = similarity_scores[max_sim_idx]
    plt.axvline(x=max_sim_idx, color='r', linestyle='--', 
                label=f'Max Similarity at frame {max_sim_idx} (score: {max_sim_value:.4f})')
    plt.legend()
    
    plt.savefig(file)

@hydra.main(version_base=None, config_path="configs", config_name="localize")
def localize(cfg: DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name
    text = cfg.text
    npy_path = cfg.npy
    window_size = cfg.window_size
    file = cfg.output_file

    import src.prepare  # noqa
    import torch
    import numpy as np
    from src.config import read_config
    from pytorch_lightning import seed_everything
    from src.data.collate import collate_x_dict

    cfg = read_config(run_dir)
    
    token_embedding_model, text_embedding_model, motion_normalization_model, motion_embedding_model = load_models(cfg, ckpt_name, device)

    seed_everything(cfg.seed)
    with torch.inference_mode():
        text_x_dict = collate_x_dict(token_embedding_model([text]))
        text_embedding = text_embedding_model.encode(text_x_dict, sample_mean=True)[0]
        text_embedding = text_embedding.cpu().numpy()

    # NOTE: the text embedding
    text_embedding = text_embedding

    motion = torch.from_numpy(np.load(npy_path)).to(torch.float)
    motion = motion_normalization_model(motion)
    motion = motion.to(device)
    
    motion_length = len(motion)
    # NOTE: controls the granularity
    stride = 1
    
    number_of_windows = max(1, motion_length - window_size + 1)
    similarity_scores = np.zeros(number_of_windows)
    
    logger.info(f"Processing {number_of_windows} windows with window size {window_size}")
    
    for i in range(0, number_of_windows, stride):
        window_start = i
        window_end = i + window_size
        
        if window_end > motion_length:
            window_end = motion_length
        
        motion_window = motion[window_start:window_end]
        
        motion_window_dict = {"x": motion_window, "length": len(motion_window)}
        
        seed_everything(cfg.seed)
        with torch.inference_mode():
            window_x_dict = collate_x_dict([motion_window_dict])
            motion_window_embedding = motion_embedding_model.encode(window_x_dict, sample_mean=True)[0]
            motion_window_embedding = motion_window_embedding.cpu().numpy()
        
        similarity = compute_similarity(text_embedding, motion_window_embedding)
        similarity_scores[i] = similarity
    
    logger.info(f"Similarity scores shaoe: {similarity_scores.shape}")
    logger.info(f"Max similarity score: {similarity_scores.max()}")
    logger.info(f"Max similarity index: {similarity_scores.argmax()}")
    
    # --- --- ---

    visualize_results(file, text, similarity_scores)

if __name__ == "__main__":
    localize()