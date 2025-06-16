import torch
from .index import LabelExtractor

class MajorityBasedStartEndWithMajority(LabelExtractor):
    def __init__(self):
        pass

    def extract(
        self,
        # [B, T], with T being the window size
        transition_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T = transition_mask.shape

        result = torch.full((B, 3), -1.0, device=transition_mask.device)

        for i in range(B):
            # NOTE: [T]
            mask = transition_mask[i]
            
            unique_classes, counts = mask.unique(return_counts=True)
            majority_class = unique_classes[torch.argmax(counts)]
            
            result[i, 0] = majority_class.float()

            indices = torch.nonzero(mask == majority_class, as_tuple=False).view(-1)

            if indices.numel() > 0:
                start = indices[0].item() / (T - 1)
                end = indices[-1].item() / (T - 1)
                result[i, 1] = start
                result[i, 2] = end

        return result