import torch

from abc import ABC, abstractmethod

class LabelExtractor(ABC):
    
    @abstractmethod
    def extract(
        self,
        annotation: torch.Tensor,
    )-> torch.Tensor:
        """
        Receives a batch of transition masks and extracts the labels from it.
        
        Args:
            annotation (torch.Tensor): A tensor of shape (B, T) where B is the batch size and T is the number of frames.
            
        Returns:
            torch.Tensor: A tensor of shape (B, 3) where each row contains:
                - class (integer index)
                - start_transition (0-1 or -1)
                - end_transition (0-1 or -1)
        """
        pass