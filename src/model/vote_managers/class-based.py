import torch

from .index import VoteManager

class ClassBasedVoteManager(VoteManager):
    def __init__(self):
        pass
    
    def aggregate(
        self,
        windows_scores: torch.Tensor,
        number_of_frames: int,
        window_size: int,
        window_step: int = 1
    ):
        # NOTE: windows_scores is a tensor of shape (T_new,)
        # For each window, we are given the score of the window
        
        device = windows_scores.device
        
        number_of_windows = windows_scores.shape[0]
        # total_number_of_frames = (number_of_windows - 1) * window_step + window_size
        total_number_of_frames = number_of_frames
        
        votes = torch.zeros(total_number_of_frames, device=device)
        counts = torch.zeros(total_number_of_frames, device=device)
        
        for i in range(number_of_windows):
            start = i * window_step
            end = start + window_size
            
            if end > total_number_of_frames:
                break
            
            window_class = (torch.sigmoid(windows_scores[i]) > 0.5).float()
            
            votes[start:end] += window_class
            counts[start:end] += 1.0
            
        proportions = votes / counts

        # NOTE: majority vote: if proportion > 0.5, predict class 1, else 0
        per_frame_classes: torch.Tensor = (proportions > 0.5).float()
        
        return per_frame_classes