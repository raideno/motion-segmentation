import gc
import pdb
import torch
import logging
import torch.nn as nn

from typing import Dict
from torch import Tensor
from pytorch_lightning import LightningModule

from src.model.vote_managers.index import VoteManager
from src.model.label_extractors.index import LabelExtractor

logger = logging.getLogger(__name__)

class StartEndSegmentationModel(LightningModule):
    def __init__(
        self,
        motion_encoder: nn.Module,
        classifier: nn.Module,
        label_extractor: LabelExtractor,
        lr: float,
        window_positional_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.lr = lr
        self.lmd = {}
        
        self.label_extractor = label_extractor
        self.motion_encoder = motion_encoder
        self.classifier = classifier
        self.window_positional_encoder = window_positional_encoder

        self.classification_loss_fn = nn.BCEWithLogitsLoss()
        self.start_regression_loss_fn = nn.MSELoss()
        self.end_regression_loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # def forward(self, batch, batch_index):
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = args[0]
        batch_index = kwargs.get("batch_index", 0) 
        
        preprocessed_motion = batch["transformed_motion"]
        motion = batch["motion"]
        transition_mask = batch["annotation"]
        
        window_position = batch.get("window_position", None)
        # NOTE: number of windows in the sequence
        sequence_size = batch.get("sequence_size", None)
        
        latent = self.motion_encoder(batch)
        
        if self.window_positional_encoder is not None and window_position is not None and sequence_size is not None:
            positional_embedding = self.window_positional_encoder(preprocessed_motion, sequence_size)
            latent = positional_embedding + positional_embedding
        
        class_logits, start_logits, end_logits = self.classifier(latent)
        
        return class_logits, start_logits, end_logits
    
    def compute_loss(self, class_logits: torch.Tensor, start_logits: torch.Tensor, end_logits: torch.Tensor, label: torch.Tensor):
        # pdb.set_trace()
        
        class_loss: torch.Tensor = self.classification_loss_fn(class_logits, label[:, 0])

        valid_start_mask = label[:, 1] != -1
        valid_end_mask = label[:, 2] != -1

        if valid_start_mask.any():
            start_loss: torch.Tensor = self.start_regression_loss_fn(start_logits[valid_start_mask], label[:, 1][valid_start_mask])
        else:
            start_loss = torch.tensor(0.0, device=label.device)

        if valid_end_mask.any():
            end_loss: torch.Tensor = self.end_regression_loss_fn(end_logits[valid_end_mask], label[:, 2][valid_end_mask])
        else:
            end_loss = torch.tensor(0.0, device=label.device)

        loss = class_loss + start_loss + end_loss

        return loss, (class_loss, start_loss, end_loss)

    def step(self, batch, batch_idx):
        # NOTE: preprocessed_motion (B, window_size, 263)
        # NOTE: motion (B, WINDOW_SIZE, 22, 3)
        # NOTE: transition_mask (B,)
        # preprocessed_motion, motion, transition_mask = batch
        
        preprocessed_motion = batch["transformed_motion"]
        motion = batch["motion"]
        transition_mask = batch["annotation"]
        
        # pdb.set_trace()
        
        label = self.label_extractor.extract(transition_mask)
                
        class_logits, start_logits, end_logits = self.forward(batch, batch_idx)
        
        class_logits = class_logits.view(-1)
        start_logits = start_logits.view(-1)
        end_logits = end_logits.view(-1)
    
        loss, (class_loss, start_loss, end_loss) = self.compute_loss(class_logits, start_logits, end_logits, label)
        
        return loss, (class_loss, start_loss, end_loss)
    
    def training_step(self, *args, **kwargs) :
        batch = args[0]
        batch_idx = kwargs.get("batch_idx", 0)
        
        # logger.info(f"Before forward: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        loss, (class_loss, start_loss, end_loss) = self.step(batch, batch_idx)
        # logger.info(f"After forward / step: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_class_loss", class_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_start_loss", start_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_end_loss", end_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def training_step(self, *args, **kwargs) :
        batch = args[0]
        batch_idx = kwargs.get("batch_idx", 0)
        
        loss, (class_loss, start_loss, end_loss) = self.step(batch, batch_idx)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_class_loss", class_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_start_loss", start_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_end_loss", end_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def segment_sequence(
        self,
        batch: Dict,
        window_size: int,
        vote_manager: VoteManager,
        window_step: int = 1,
        mean = None,
        std = None,
    ) -> tuple[Tensor | None, Exception | None]:
        """
        Segment a motion sequence using sliding window classification.

        Args:
            batch: Dictionary containing keys 'transformed_motion', 'motion', and 'transition_mask'
            window_size: Size of the sliding window
            window_step: Step size for sliding the window
            vote_manager: Instance of VoteManager to aggregate window scores

        Returns:
            Tensor: Per-frame classifications of shape (T,) where 1 indicates transition frame and 0 indicates action frame
        """
        motion_sequence = batch["transformed_motion"]  # (T, 22, 3)
        transformed_motion_sequence = batch["motion"]  # (T, 263)
        
        T, _, _ = motion_sequence.shape
        
        device = motion_sequence.device

        if T < window_size:
            logger.warning(f"Short sequence (T={T}) padded to window size ({window_size})")
            return None, Exception("Sequence too short for window size")

        transformed_motion_windows = transformed_motion_sequence.unfold(dimension=0, size=window_size, step=window_step)
        motion_windows = motion_sequence.unfold(dimension=0, size=window_size, step=window_step)
        
        # NOTE: unfold creates a new dimension for the window, so we need to permute the dimensions
        transformed_motion_windows = transformed_motion_windows.permute(0, 2, 1)  # (T_new, D, W)
        motion_windows = motion_windows.permute(0, 3, 1, 2) # (T_new, W, 22, 3)
        
        T_new, W, _, _ = motion_windows.shape
        
        if mean is not None and std is not None:
            mean = mean.to(device)
            std = std.to(device)
            motion_windows = (motion_windows - mean) / (std + 1e-8)

        window_batch = {
            "transformed_motion": transformed_motion_windows.to(device).float(),
            "motion": motion_windows.to(device).float(),
            # NOTE: dummy
            "transition_mask": torch.zeros(T_new, W, device=device).float()
        }
        
        window_batch["annotation"] = window_batch["transition_mask"]
        
        self.eval()
        with torch.no_grad():
            # NOTE: (T_new,)
            class_logits, _, _ = self.forward(window_batch, None)
            class_logits = class_logits.view(T_new)
            
        per_frame_classes = vote_manager.aggregate(
            windows_scores=class_logits,
            number_of_frames=T,
            window_size=window_size,
            window_step=window_step
        )

        return per_frame_classes, None
        
    def on_epoch_end(self):
        torch.cuda.empty_cache()
        import gc
        gc.collect()