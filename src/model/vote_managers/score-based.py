import torch

from .index import VoteManager

class ScoreBasedVoteManager(VoteManager):
    def __init__(self):
        pass
    
    def aggregate(
        self,
        windows_scores: torch.Tensor,
        number_of_frames: int,
        window_size: int,
        window_step: int = 1,
    ):
        # NOTE: windows_scores is a tensor of shape (T_new,)
        # For each window, we are given the score of the window
        
        device = windows_scores.device
        
        number_of_windows = windows_scores.shape[0]
        # total_number_of_frames = window_size * number_of_windows
        # total_number_of_frames = (number_of_windows - 1) * window_step + window_size
        total_number_of_frames = number_of_frames
        
        votes = torch.zeros(total_number_of_frames, device=device)
        counts = torch.zeros(total_number_of_frames, device=device)
        
        for i in range(number_of_windows):
            start = i * window_step
            end = start + window_size
            
            if end > total_number_of_frames:
                break
            
            votes[start:end] += windows_scores[i]
            counts[start:end] += 1.0
            
        # NOTE: avoid division by zero, even tho it should not happen as we'll be having at least one vote per frame
        counts = counts.clamp(min=1.0)
        
        per_frame_logits = votes / counts
        
        per_frame_probabilities = torch.sigmoid(per_frame_logits)
        
        per_frame_classes = (per_frame_probabilities > 0.5).float()
        
        return per_frame_classes