# NOTE: python encode_sub_sequences.py output_dir=/home/nadir/disk/code/sub-sequences-embeddings texts_dir=/home/nadir/disk/code/texts-sub-sequences

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

@hydra.main(version_base=None, config_path="configs", config_name="encode_sub_sequences")
def encode_sub_sequences(cfg: DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name
    
    texts_dir = cfg.texts_dir
    output_dir = cfg.output_dir
    
    batch_size = int(cfg.batch_size)
    
    os.makedirs(output_dir, exist_ok=True)

    import tqdm
    import torch
    import src.prepare # noqa
    
    from src.config import read_config
    from pytorch_lightning import seed_everything
    from src.data.collate import collate_x_dict

    cfg = read_config(run_dir)
    
    token_embedding_model, text_embedding_model, motion_normalization_model, motion_embedding_model = load_models(cfg, ckpt_name, device)
    
    files_names = [filename for filename in os.listdir(texts_dir) if filename.endswith(".txt")]
    
    seed_everything(cfg.seed, verbose=False)
    with tqdm.tqdm(iterable=enumerate(files_names), total=len(files_names), desc="[processing]") as progress_bar:
        for i, texts_filename in progress_bar:
            try:
                txt_path = os.path.join(texts_dir, texts_filename)
                
                with open(txt_path, "r") as file:
                    texts = file.readlines()
                    
                texts = [text.strip() for text in texts]
                
                with torch.inference_mode():
                    # text_x_dict = collate_x_dict(token_embedding_model(text))
                    text_x_dict = collate_x_dict(token_embedding_model(texts))
                    latent = text_embedding_model.encode(text_x_dict, sample_mean=True)
                    latent = latent.cpu().numpy()

                output_path = os.path.join(output_dir, f"{texts_filename.split('.')[0]}_embeddings.npy")
                
                np.save(output_path, latent)
            except Exception as exception:
                logger.error(f"Error processing {texts_filename}: {exception}")
                logger.debug(traceback.format_exc())
                continue
        
if __name__ == "__main__":
    encode_sub_sequences()