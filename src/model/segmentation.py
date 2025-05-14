import gc
import torch
import logging
import torch.nn as nn

from typing import Dict
from torch import Tensor
from pytorch_lightning import LightningModule

logger = logging.getLogger(__name__)

class SegmentationModel(LightningModule):
    def __init__(
        self,
        motion_encoder: nn.Module,
        window_size: int,
        window_step: int,
        hidden_dim: int,
        lr: float,
    ):
        super().__init__()
        self.motion_encoder = motion_encoder
        self.window_size = window_size
        self.window_step = window_step
        self.lr = lr
        self.lmd = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5}

        self.motion_encoder.eval()
        for param in self.motion_encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.motion_encoder.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, motion_x_dict: Dict) -> Tensor:
        motion_sequence = motion_x_dict["x"]
        original_mask = motion_x_dict["mask"]
        
        B, T, D = motion_sequence.shape
        
        # NOTE: padding of motions that are smaller than window_size
        if T < self.window_size:
            pad_len = self.window_size - T
            motion_sequence = torch.nn.functional.pad(motion_sequence, (0, 0, 0, pad_len), mode="constant", value=0)
            original_mask = torch.nn.functional.pad(original_mask, (0, pad_len), mode="constant", value=0)
            T = motion_sequence.shape[1]
            
            logger.warning(f"Short sequence (T={T}) padded to window size ({self.window_size})")

        # NOTE: creation of windows
        windows = motion_sequence.unfold(dimension=1, size=self.window_size, step=self.window_step)
        window_masks = original_mask.unfold(dimension=1, size=self.window_size, step=self.window_step)

        B, T_new, D, W = windows.shape

        windows = windows.reshape(B * T_new, W, D)
        window_masks = window_masks.reshape(B * T_new, W)

        window_x_dict = {
            "x": windows,
            "mask": window_masks,
        }
        
        # logger.info(f"[motion_encoder.device]: {next(self.motion_encoder.parameters()).device}")
        # logger.info(f"[motion_sequence.device]: {motion_sequence.device}")
        # logger.info(f"[x.device]: {window_x_dict['x'].device}")
        # logger.info(f"[mask.device]: {window_x_dict['mask'].device}")

        with torch.no_grad():
            # NOTE: (B*T_new, 1, latent_dim)
            encoded = self.motion_encoder(window_x_dict)
            # NOTE: we get rid of the "1" dimension
            latent = encoded[:, 0]
            del encoded

        # NOTE: (B*T_new, 1)
        logits = self.classifier(latent)
        # NOTE: (B, T_new)
        logits = logits.view(B, T_new)

        # NOTE: we apply the mask to the logits, to avoid using padded windows
        valid_windows = window_masks.all(dim=1).view(B, T_new)
        logits[~valid_windows] = -100.0

        votes = torch.zeros(B, T, device=logits.device)
        counts = torch.zeros(B, T, device=logits.device)

        for i in range(T_new):
            window_start = i * self.window_step
            window_end = window_start + self.window_size

            if window_end > T:
                break  # handle edge case for small paddings

            votes[:, window_start:window_end] += logits[:, i].unsqueeze(1)
            counts[:, window_start:window_end] += 1.0

        counts = torch.clamp(counts, min=1.0)

        # NOTE: per frame scores
        per_frame_logits = votes / counts
        
        del votes, counts, logits, valid_windows, window_masks, windows
        torch.cuda.empty_cache()
        gc.collect()

        # NOTE: (B, T)
        return per_frame_logits
    
    def compute_loss(self, logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        loss = self.loss_fn(logits[mask], targets[mask])
        return loss

    # TODO: change, rather than setting to 1, set it proportional to the size of the overlap
    def get_labels_for_windows(self, batch_segments, window_counts):
        batch_size = len(batch_segments)
        
        max_windows = max(window_counts)
        
        batch_labels = torch.zeros((batch_size, max_windows), dtype=torch.float, device=self.device)
        
        for b, (segments, num_windows) in enumerate(zip(batch_segments, window_counts)):
            for segment in segments:
                if segment["label"] == 1:
                    start = segment["start"]
                    end = segment["end"]
                    
                    for i in range(num_windows):
                        window_start = i * self.window_step
                        window_end = window_start + self.window_size
                        
                        # NOTE: compute overlap
                        overlap = max(0, min(window_end, end) - max(window_start, start))
                        
                        if overlap > 0:
                            batch_labels[b, i] = 1
        
        return batch_labels
    
    def step(self, batch, batch_idx) -> Tensor:
        motion_x_dict = batch["motion_x_dict"]
        targets = batch["transition_mask"]

        # NOTE (B, T)
        logits = self(motion_x_dict)
        # NOTE: (B, T)
        mask = motion_x_dict["mask"].bool()

        loss = self.compute_loss(logits, targets, mask)
        
        return logits, targets, loss
    
    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # logger.info(f"Before forward: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logits, targets, loss = self.step(batch, batch_idx)
        # logger.info(f"After forward / step: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        logits, targets, loss = self.step(batch, batch_idx)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def on_epoch_end(self):
        torch.cuda.empty_cache()
        import gc
        gc.collect()