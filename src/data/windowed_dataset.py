import os
import glob
import tqdm
import torch
import random

import numpy as np

from torch.utils.data import Dataset

class WindowedDataset(Dataset):
    def __init__(
        self,
        dir,
        window_size,
        window_step=1,
        preload=True,
        tiny=-1,
        split="all",
        balanced=False,
        normalize=False,
    ):
        self.window_size = window_size
        self.window_step = window_step
        self.dir = os.path.join(dir, f"{self.window_size}-{self.window_step}")
        self.tiny = tiny
        self.split = split
        self.preload = preload
        self.balanced = balanced
        self.normalize = normalize
        
        self.motion_mean = None
        self.motion_std = None
        
        self.filesnames = glob.glob(f"**/*.*.pt", root_dir=self.dir, recursive=True)
        self.filesnames = sorted(self.filesnames, key=lambda x: int(os.path.basename(x).split(".")[0]))
            
        self._apply_split()
            
        if self.balanced:
            self.filesnames = self._balance_dataset()
        
        if self.tiny > 0:
            self.tiny = min(len(self.filesnames), self.tiny)
            self.filesnames = random.sample(self.filesnames, self.tiny)
        
        if self.normalize:
            self.motion_mean, self.motion_std = self._load_statistics()
        
        if self.preload:
            for _ in tqdm.tqdm(iterable=self, total=len(self), desc="[preloading]"):
                pass
            
    def _load_statistics(self):
        statistics_path = os.path.join(self.dir, "motion_normalization_stats.pt")
        
        statistics = torch.load(statistics_path)
        
        return statistics["mean"], statistics["std"]
            
    def _apply_split(self):
        rng = np.random.default_rng(seed=42)
        
        files = np.array(self.filesnames)
        indices = np.arange(len(files))
        rng.shuffle(indices)

        train_end = int(0.8 * len(files))
        val_end = int(0.9 * len(files))

        if self.split == "train":
            self.filesnames = files[indices[:train_end]].tolist()
        elif self.split == "val":
            self.filesnames = files[indices[train_end:val_end]].tolist()
        elif self.split == "test":
            self.filesnames = files[indices[val_end:]].tolist()
        elif self.split == "all":
            self.filesnames = files.tolist()
        else:
            raise ValueError(f"Unknown split '{self.split}'. Choose from 'all', 'train', 'val', or 'test'.")
            
    def _balance_dataset(self):
        self.class_0_files = []
        self.class_1_files = []
        
        for filename in self.filesnames:
            if ".False." in filename:
                self.class_0_files.append(filename)
            elif ".True." in filename:
                self.class_1_files.append(filename)
        
        min_count = min(len(self.class_0_files), len(self.class_1_files))
        
        balanced_class_0 = random.sample(self.class_0_files, min_count)
        balanced_class_1 = random.sample(self.class_1_files, min_count)
        
        self.filesnames = balanced_class_0 + balanced_class_1
        
        random.shuffle(self.filesnames)
        
        return self.filesnames
        
    def __len__(self):
        return len(self.filesnames)
    
    def __getitem__(self, idx):
        filename = self.filesnames[idx]
        
        path = os.path.join(self.dir, filename)
        
        data = torch.load(path, weights_only=False, map_location="cpu")
        
        preprocessed_motion = data["transformed_motion"]
        motion = data["motion"]
        transition_mask = data["transition_mask"]
        
        if self.normalize and self.motion_mean is not None and self.motion_std is not None:
            if not isinstance(motion, torch.Tensor):
                motion = torch.tensor(motion)
            
            motion = (motion - self.motion_mean) / (self.motion_std + 1e-8)
            
            data["motion"] = motion
        
        return data