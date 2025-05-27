import torch

from .stgcn_helpers import STGCN

class CachedTMR(torch.nn.Module):
    def __init__(
        self,
        latent_dim=256
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
    def forward(self, batch):
        motion, label, embedding = batch
        
        return embedding