# NOTE: to start training
# python train-segmentation.py +data=babel-segmentation +model=segmentation

import tqdm
import hydra
import torch
import logging

import pytorch_lightning as lightning

from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config

# --- --- --- --- --- --- ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
# --- --- --- --- --- --- ---

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="train-segmentation", version_base="1.3")
def train_segmentation(cfg: DictConfig):
    ckpt = None
    if cfg.resume_dir is not None:
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{cfg.resume_dir}")
    else:
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    logger.info(f"[ckpt]: {ckpt}")

    lightning.seed_everything(cfg.seed)

    logger.info("[data]: loading the dataloaders")
    
    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")
    
    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )
    
    for sample in tqdm.tqdm(train_dataloader, desc="[train-dataloader-looping]"):
        continue
    
    for sample in tqdm.tqdm(val_dataloader, desc="[val-dataloader-looping]"):
        continue
    
    logger.info("[model]: loading the model")
    model = instantiate(cfg.model)
    
    logger.info("[model]: loading motion encoder weights")
    
    model.motion_encoder.load_state_dict(
        torch.load("/home/nadir/disk/codes/tmr-code/models/tmr_humanml3d_guoh3dfeats/last_weights/motion_encoder.pt")
    )

    trainer = instantiate(cfg.trainer)
    
    logger.info("[training]: started")    
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)

if __name__ == "__main__":
    train_segmentation()
