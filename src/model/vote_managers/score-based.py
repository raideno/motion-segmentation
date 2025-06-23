import torch

from .index import VoteManager

class ScoreBasedVoteManager(VoteManager):
    def __init__(self):
        pass
    
    def aggregate(
        self,
        windows_scores: torch.Tensor,  # shape: (T_new, num_classes)
        number_of_frames: int,
        window_size: int,
        window_step: int = 1,
    ):
        device = windows_scores.device
        number_of_windows, num_classes = windows_scores.shape
        
        votes = torch.zeros((number_of_frames, num_classes), device=device)
        counts = torch.zeros(number_of_frames, device=device)
        
        for i in range(number_of_windows):
            start = i * window_step
            end = start + window_size
            
            if end > number_of_frames:
                break
            
            # Add the logits vector to each frame in the window
            # votes[start:end] shape: (window_size, num_classes)
            # windows_scores[i] shape: (num_classes,)
            votes[start:end] += windows_scores[i].unsqueeze(0).expand(end - start, -1)
            
            counts[start:end] += 1.0
        
        counts = counts.clamp(min=1.0).unsqueeze(1)  # shape: (number_of_frames, 1)
        
        # Average logits per frame per class
        per_frame_logits = votes / counts
        
        # Convert logits to probabilities using softmax across classes for each frame
        per_frame_probabilities = torch.softmax(per_frame_logits, dim=1)  # (number_of_frames, num_classes)
        
        # Pick the class with the highest probability for each frame
        per_frame_classes = torch.argmax(per_frame_probabilities, dim=1)  # (number_of_frames,)
        
        return per_frame_classes
    
    def aggregate_logits(
        self,
        windows_scores: torch.Tensor,  
        number_of_frames: int,
        window_size: int,
        window_step: int = 1,
        apply_softmax: bool = False,
    ):
        device = windows_scores.device
        number_of_windows, num_classes = windows_scores.shape

        votes = torch.zeros((number_of_frames, num_classes), device=device)
        counts = torch.zeros(number_of_frames, device=device)

        for i in range(number_of_windows):
            start = i * window_step
            end = start + window_size

            if end > number_of_frames:
                break

            votes[start:end] += windows_scores[i].unsqueeze(0).expand(end - start, -1)
            counts[start:end] += 1.0

        counts = counts.clamp(min=1.0).unsqueeze(1)
        per_frame_logits = votes / counts

        if apply_softmax:
            return torch.softmax(per_frame_logits, dim=1)  # shape: (number_of_frames, num_classes)
        else:
            return per_frame_logits 