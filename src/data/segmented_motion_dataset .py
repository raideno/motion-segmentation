import torch

from .text_motion import TextMotionDataset

def collate_segmented_motion(batch):
    motion_sequences = [item["motion"] for item in batch]
    lengths = [item["length"] for item in batch]
    
    # max_length = max(lengths)
    window_size = 30
    max_length = max(window_size, max(lengths))
    
    batch_size = len(motion_sequences)
    
    feature_dimension = motion_sequences[0].shape[-1]
    
    padded_motion = torch.zeros((batch_size, max_length, feature_dimension), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    
    for i, (seq, length) in enumerate(zip(motion_sequences, lengths)):
        padded_motion[i, :length] = seq
        mask[i, :length] = True
    
    texts = [item["text"] for item in batch]
    keyids = [item["keyid"] for item in batch]
    segments = [item["segments"] for item in batch]
    
    return {
        "motion_x_dict": {
            "x": padded_motion,
            "length": lengths,
            "mask": mask
        },
        "text": texts,
        "keyid": keyids,
        "segments": segments
    }

class SegmentedMotionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        motion_loader,
        text_to_sent_emb,
        text_to_token_emb,
        split: str = "train",
        min_seconds: float = 0.0,
        max_seconds: float = 100.0,
        preload: bool = True,
        tiny: bool = False
    ):
        self.base_dataset = TextMotionDataset(
            path,
            motion_loader,
            text_to_sent_emb,
            text_to_token_emb,
            split,
            min_seconds,
            max_seconds,
            preload,
            tiny,
        )
        self.collate_fn = collate_segmented_motion

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        motion = item["motion_x_dict"]["x"]
        text = item["text"]
        
        T = motion.shape[0]
        
        label = 1 if "transition" in text.lower() else 0
        segments = [{"start": 0, "end": T, "label": label}]
        
        return {
            "motion": motion,
            # NOTE: the length is required for mask creation
            "length": item["motion_x_dict"]["length"],
            "segments": segments,
            "text": text,
            "keyid": item["keyid"],
        }