import torch
import torch.nn as nn

class NonePositionalEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        
        self.d_model = d_model

    def forward(self, batch) -> torch.Tensor:
        """
        Forward pass for the positional encoder.
        
        Args:
            batch: A dictionary of the following structure.
            {
                "motion": (batch-size, window-size, 22, 3),
                "transformed_motion":(batch-size, window-size, 263),
                "annotation": (batch-size, window-size, 1),
                # NOTE: 
                "window_position": (batch-size, 1),
                "sequence_size": (batch-size, 1),
            }
       
        Returns:
            Tensor of zeros of shape (batch_size, d_model)
        """
        device = batch["motion"].device
        batch_size = batch["motion"].shape[0]
        return torch.zeros(batch_size, self.d_model, device=device)