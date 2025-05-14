import torch
import random

def collate_segmented_motion(batch):
    motion_sequences = [item["motion"] for item in batch]
    lengths = [item["length"] for item in batch]
    
    max_length = max(lengths)
    
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

class DummySegmentedMotionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10000, seq_len=32, feature_dim=263):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.collate_fn = collate_segmented_motion  # use your real collate fn

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        length = self.seq_len
        # length = random.randint(int(self.seq_len * 0.5), self.seq_len)
        motion = torch.randn(length, self.feature_dim)
        text = "This is a dummy motion sample."
        label = random.randint(0, 1)
        segments = [{"start": 0, "end": length, "label": label}]
        keyid = str(idx)

        return {
            "motion": motion,
            "length": length,
            "segments": segments,
            "text": text,
            "keyid": keyid,
        }