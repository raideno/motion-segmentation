import torch
import torch.nn as nn

class IntegerBasedWindowPositionalEncoder(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super(IntegerBasedWindowPositionalEncoder, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        self.pos_embedding = nn.Embedding(max_len, d_model)

    # def forward(self, window_position: torch.Tensor, window_size: torch.Tensor) -> torch.Tensor:
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
        window_position = batch['window_position']
        
        return self.pos_embedding(window_position)