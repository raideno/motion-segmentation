import torch
import torch.nn as nn

class RelativeWindowPositionalEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

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
            Tensor of shape (batch_size, d_model)
        """
        # NOTE: (B, 1)
        window_position = batch["window_position"].float()
        # NOTE: (B, 1)
        sequence_length_in_frames = batch["sequence_size"].float()

        window_size = batch["motion"].shape[1]

        # NOTE: calculate total number of windows per sequence; (B, 1)
        total_windows = sequence_length_in_frames / window_size

        denom = (total_windows - 1).clamp(min=1.0)

        # NOTE: normalized [0,1]
        # shape [32]
        relative_pos = window_position / denom
        
        # shape [32, 1]
        relative_pos = relative_pos.unsqueeze(-1)

        return self.proj(relative_pos)