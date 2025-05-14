# NOTE: python encode_motions_windows.py output_dir=/home/nadir/disk/code/motions-windows-embeddings motions_dir=/home/nadir/disk/codes/human-ml3d-code/HumanML3D/new_joint_vecs window_size=20

import os
import hydra
import logging
import traceback

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

@hydra.main(version_base=None, config_path="configs", config_name="encode_motions_windows")
def encode_motions_windows(cfg: DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name
    
    motions_dir = cfg.motions_dir
    output_dir = cfg.output_dir
    
    stride = int(cfg.stride)
    batch_size = int(cfg.batch_size)
    window_size = int(cfg.window_size)
    
    os.makedirs(output_dir, exist_ok=True)

    import tqdm
    import torch
    import src.prepare  # noqa
    
    from src.config import read_config
    from pytorch_lightning import seed_everything
    from src.data.collate import collate_x_dict

    cfg = read_config(run_dir)
    
    token_embedding_model, text_embedding_model, motion_normalization_model, motion_embedding_model = load_models(cfg, ckpt_name, device)
    
    files_names = [filename for filename in os.listdir(motions_dir) if filename.endswith(".npy")]
    
    seed_everything(cfg.seed, verbose=False)
    with tqdm.tqdm(iterable=enumerate(files_names), total=len(files_names), desc="[processing]") as progress_bar:
        for i, motion_filename in progress_bar:
            try:
                npy_path = os.path.join(motions_dir, motion_filename)

                motion = torch.from_numpy(np.load(npy_path)).to(torch.float)
                
                motion = motion_normalization_model(motion)
                motion = motion.to(device)
        
                motion_length = len(motion)
                # NOTE: controls the granularity
        
                number_of_windows = max(1, motion_length - window_size + 1)
                
                motion_windows_embeddings = []
        
                motion_window_dicts = []

                for i in range(0, number_of_windows, stride):
                    window_start = i
                    window_end = i + window_size
                    if window_end > motion_length:
                        window_end = motion_length
                    motion_window = motion[window_start:window_end]
                    motion_window_dicts.append({"x": motion_window, "length": len(motion_window)})
                    
                with torch.inference_mode():
                    window_x_dict = collate_x_dict(motion_window_dicts)
                    motion_windows_embeddings = motion_embedding_model.encode(window_x_dict, sample_mean=True)
                    motion_windows_embeddings = motion_windows_embeddings.cpu().numpy()
                
                output_path = os.path.join(output_dir, f"{motion_filename.split('.')[0]}_windows.npy")
                
                np.save(output_path, motion_windows_embeddings)
            except Exception as exception:
                logger.error(f"Error processing {motion_filename}: {exception}")
                logger.debug(traceback.format_exc())
                continue
        
if __name__ == "__main__":
    encode_motions_windows()