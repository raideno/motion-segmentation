import torch

from .index import LabelExtractor

class TransitionBasedStartEndWithoutMajority(LabelExtractor):
    def __init__(self):
        pass
    
    def extract(
        self,
        transition_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T = transition_mask.shape
        result = torch.full((B, 3), -1.0, device=transition_mask.device)

        for i in range(B):
            mask = transition_mask[i]
            indices = torch.nonzero(mask, as_tuple=False).view(-1)
            if indices.numel() > 0:
                start = indices[0].item() / (T - 1)
                end = indices[-1].item() / (T - 1)
                result[i, 0] = 1.0
                result[i, 1] = start
                result[i, 2] = end
            else:
                result[i, 0] = 0.0

        return result