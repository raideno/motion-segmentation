# HYDRA_FULL_ERROR=1 python train-start-end-segmentation.py --multirun \
#   ++data.window_size=10,15,20,25,30 \
#   ++model.motion_encoder.pretrained=false,true \
#   ++data.dir=/home/nadir/windowed-babel/ \
#   ++data.balanced=true \
#   ++data.normalize=true \
#   model/label_extractor=majority-based-start-end-with-majority,transition-based-start-end-with-majority,transition-based-start-end-without-majority \
#   model/motion_encoder=tmr \
#   model/classifier=mlp

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

@hydra.main(config_path="configs", config_name="train-start-end-segmentation", version_base="1.3")
def train_start_end_segmentation(cfg: DictConfig):
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

    logger.info(f"[cfg.run_dir]: {cfg.run_dir}")
    logger.info(f"[cfg.data]: {cfg.data}")
    logger.info(f"[cfg.model.classifier]: {cfg.model.classifier}")
    logger.info(f"[cfg.model.motion_encoder]: {cfg.model.motion_encoder}")

    logger.info(f"[ckpt]: {ckpt}")

    lightning.seed_everything(cfg.seed)

    logger.info("[data]: loading the dataloaders")
    
    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")
    
    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        # collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        # collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )
    
    logger.info("[model]: loading the model")
    model = instantiate(cfg.model)
    
    logger.info("[model]: loading motion encoder weights")
    
    trainer = instantiate(cfg.trainer)
    
    logger.info("[training]: started")    
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)

if __name__ == "__main__":
    # NOTE: issue was that the dataset was the one moving tensors to gpu and thus CUDA was involved before dataloader
    # i modified it and set it to move to CPU and dataloader is the one responsible of moving to GPU, more specifically pytorch lightinig
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    
    train_start_end_segmentation()
