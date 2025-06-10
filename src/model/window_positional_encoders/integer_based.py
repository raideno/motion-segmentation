import torch
import torch.nn as nn

class IntegerBasedWindowPositionalEncoder(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super(IntegerBasedWindowPositionalEncoder, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, window_position: torch.Tensor, window_size: torch.Tensor) -> torch.Tensor:
        """
        window_position: Tensor of shape (batch_size,)
                containing integer window_position indices
        window_size: Tensor of shape (batch_size,)
            containing integer window_sizes, number of windows in the sequence
        Returns:
            Tensor of shape (batch_size, d_model)
        """
        return self.pos_embedding(window_position)