import torch

from abc import ABC, abstractmethod

class VoteManager(ABC):
    
    @abstractmethod
    def aggregate(
        self,
        number_of_frames: int,
        windows_scores: torch.Tensor,
        window_size: int,
        window_step: int = 1
    )-> torch.Tensor:
        """Abstract method to aggregate votes"""
        pass