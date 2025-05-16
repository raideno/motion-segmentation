import gc
import json
import torch
import logging

import numpy as np
import torch.nn as nn

from typing import Dict
from torch import Tensor
from pytorch_lightning import LightningModule

logger = logging.getLogger(__name__)

class ClassifierModel(LightningModule):
    def __init__(
        self,
        motion_encoder: nn.Module,
        hidden_dim: int,
        lr: float,
        cache: bool = True
    ):
        super().__init__()
        self.motion_encoder = motion_encoder
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        self.lmd = { }
        
        self.cache = cache
        self.cached_latents, self.keyIdsToIndex = self._setup_caching()
    
        self.motion_encoder.eval()
        for param in self.motion_encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.motion_encoder.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def _setup_caching(self):
        JSON = "/home/nadir/disk/codes/tmr-code/models/tmr_humanml3d_guoh3dfeats/latents/babel_keyids_index_all.json"
        CACHE = "/home/nadir/disk/codes/tmr-code/models/tmr_humanml3d_guoh3dfeats/latents/babel_all.npy"
        
        keyidsToIndex = json.load(open(JSON, "rb"))
        cached_latents = np.load(CACHE)
        
        return cached_latents, keyidsToIndex

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def get_latent(self, batch, batch_index) -> Tensor:
        if self.cache and batch_index is not None:
            keyids = batch["keyid"]
            
            latent = [self.cached_latents[self.keyIdsToIndex[keyid]] for keyid in keyids]
            latent = torch.tensor(latent)
        else:
            with torch.no_grad():
                # NOTE: (B, 1, latent_dim)
                # encoded = self.motion_encoder(motion_x_dict)
                encoded = self.motion_encoder(batch["motion_x_dict"])
                # NOTE: we get rid of the "1" dimension
                latent = encoded[:, 0]
                del encoded
                    
        return latent
                    
    def forward(self, batch: Dict, batch_index) -> Tensor:
        batch_size = batch["motion_x_dict"]["x"].shape[0]
        
        latent = self.get_latent(batch, batch_index)
        
        latent = latent.to("cuda:1")

        # NOTE: (B, 1)
        logits = self.classifier(latent)
        
        # NOTE: (B,)
        logits = logits.reshape(batch_size,)

        return logits
    
    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        loss = self.loss_fn(logits, targets)
        
        return loss
    
    @staticmethod
    def get_targets(batch, batch_idx=None):
        texts = batch["text"]
        
        targets = [0 if "transition" in text else 1 for text in texts]
        targets = torch.tensor(np.array(targets), dtype=torch.float)
        
        return targets

    def step(self, batch, batch_idx=None) -> Tensor:
        # NOTE: (B, T, D)
        # motion_x_dict = batch["motion_x_dict"]
        # NOTE: (B,)
        targets = ClassifierModel.get_targets(batch, batch_idx)
        
        targets = targets.to("cuda:1")

        # NOTE (B, )
        logits = self(batch, batch_idx)

        loss = self.compute_loss(logits, targets)
        
        return logits, targets, loss
    
    def training_step(self, batch: Dict, batch_idx=None) -> Tensor:
        logits, targets, loss = self.step(batch, batch_idx)
        
        batch_size = batch["motion_x_dict"]["x"].shape[0]
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch: Dict, batch_idx=None) -> Tensor:
        logits, targets, loss = self.step(batch, batch_idx)
        
        batch_size = batch["motion_x_dict"]["x"].shape[0]
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def segment_sequence(
        self,
        motion_x_dict: Dict,
        window_size: int = 20,
        window_step: int = 5
    ) -> Tensor:
        """
        Segment a batch of motion sequences using sliding window classification.
        
        Args:
            motion_x_dict: Dictionary containing motion sequence data with keys 'x' and 'mask'
            window_size: Size of the sliding window
            window_step: Step size for sliding the window
        
        Returns:
            Tensor: Per-frame classifications of shape (B, T) where 0 indicates transition frame and 1 indicates action frame
        """
        motion_sequence = motion_x_dict["x"]
        original_mask = motion_x_dict["mask"]
        
        B, T, D = motion_sequence.shape
        device = motion_sequence.device
        
        # NOTE: padding of motions that are smaller than window_size
        if T < window_size:
            pad_len = window_size - T
            motion_sequence = torch.nn.functional.pad(motion_sequence, (0, 0, 0, pad_len), mode="constant", value=0)
            original_mask = torch.nn.functional.pad(original_mask, (0, pad_len), mode="constant", value=0)
            T = motion_sequence.shape[1]
            
            logger.warning(f"Short sequence (T={T}) padded to window size ({window_size})")

        windows = motion_sequence.unfold(dimension=1, size=window_size, step=window_step)
        window_masks = original_mask.unfold(dimension=1, size=window_size, step=window_step)

        B, T_new, D, W = windows.shape
        
        # NOTE: (B*T_new, W, D)
        # we have a batch of motions with length W
        windows = windows.reshape(B * T_new, W, D)
        window_masks = window_masks.reshape(B * T_new, W)

        window_x_dict = {
            "x": windows,
            "mask": window_masks,
            # "keyid": batch["keyid"] * T_new if "keyid" in motion_x_dict else ["unknown"] * (B * T_new)
        }
        
        # NOTE: (B * T_new)
        # we classified each motion (window) in the batch
        logits = self({ "motion_x_dict": window_x_dict, "keyid": ["unknown"] * (B * T_new) }, None)

        logits = logits.reshape(B, T_new)

        # NOTE: don't use padded windows
        # valid_windows = window_masks.all(dim=1).view(B, T_new)
        
        # print("[valid_windows.shape]:", valid_windows.shape)
        
        # logits[~valid_windows] = -100.0

        # NOTE: aggregate votes for each frame
        votes = torch.zeros(B, T, device=device)
        counts = torch.zeros(B, T, device=device)

        for i in range(T_new):
            window_start = i * window_step
            window_end = window_start + window_size

            if window_end > T:
                break

            votes[:, window_start:window_end] += logits[:, i].unsqueeze(1)
            counts[:, window_start:window_end] += 1.0

        counts = torch.clamp(counts, min=1.0)

        # NOTE: (B, T)
        per_frame_logits = votes / counts
        
        per_frame_probs = torch.sigmoid(per_frame_logits)
        per_frame_classes = (per_frame_probs > 0.5).float()
        
        # NOTE: apply dataloader padding mask, padded zones = 0
        per_frame_classes = per_frame_classes * original_mask
        
        # del votes, counts, logits, valid_windows, window_masks, windows, latent
        # del votes, counts, logits, valid_windows, window_masks, windows
        # torch.cuda.empty_cache()
        # gc.collect()

        return per_frame_classes

    def on_epoch_end(self):
        torch.cuda.empty_cache()
        import gc
        gc.collect()