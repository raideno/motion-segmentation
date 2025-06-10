import torch

from .index import LabelExtractor

class TransitionBasedStartEndWithMajority(LabelExtractor):
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
            num_ones= int(mask.sum().item())
            
            # NOTE: majority with 1 if tie
            has_majority_transition = 1.0 if num_ones >= (T / 2) else 0.0
            
            result[i, 0] = has_majority_transition

            # NOTE: if transition inside, we compute normalized start / end
            if num_ones > 0:
                indices = torch.nonzero(mask, as_tuple=False).view(-1)
                
                start = indices[0].item() / (T - 1)
                end = indices[-1].item() / (T - 1)
                
                result[i, 1] = start
                result[i, 2] = end

        return result