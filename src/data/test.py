import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
import enum

class BabelDatasetSplit(enum.IntFlag):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()
    EXTRA_TRAIN = enum.auto()
    EXTRA_VAL = enum.auto()
    ALL = TRAIN | VAL | TEST | EXTRA_TRAIN | EXTRA_VAL

SPLITS = [
    (BabelDatasetSplit.TRAIN, 'train'),
    (BabelDatasetSplit.VAL, 'val'),
    (BabelDatasetSplit.TEST, 'test'),
    (BabelDatasetSplit.EXTRA_TRAIN, 'extra_train'),
    (BabelDatasetSplit.EXTRA_VAL, 'extra_val')
]

class BabelWindowedDataset(Dataset):
    def __init__(
        self,
        babel_dir: str,
        motion_loader,  # Function to load motion data given path, start, end
        window_size: int = -1,  # -1 means no windowing (whole sequence)
        window_stride: int = 1,
        splits: BabelDatasetSplit = BabelDatasetSplit.ALL,
        fps: float = 20.0,  # Frames per second for motion data
        min_window_duration: float = 0.1,  # Minimum window duration in seconds
        prefer_frame_annotations: bool = True,  # Prefer frame over sequence annotations
        verbose: bool = False
    ):
        self.babel_dir = babel_dir
        self.motion_loader = motion_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.splits = splits
        self.fps = fps
        self.min_window_duration = min_window_duration
        self.prefer_frame_annotations = prefer_frame_annotations
        self.verbose = verbose
        
        # Load BABEL annotations
        self.raw_babel = self._load_babel_annotations()
        
        # Extract sequences and create windows
        self.sequences = self._extract_sequences()
        self.windows = self._create_windows()
        
        if self.verbose:
            print(f"Loaded {len(self.sequences)} sequences")
            print(f"Created {len(self.windows)} windows")
            if self.window_size != -1:
                print(f"Window size: {self.window_size} frames ({self.window_size/self.fps:.2f}s)")
                print(f"Window stride: {self.window_stride} frames ({self.window_stride/self.fps:.2f}s)")
    
    def _load_babel_annotations(self) -> Dict[str, Any]:
        """Load BABEL annotations from JSON files based on selected splits."""
        result = {}
        
        # Get filenames for selected splits
        split_filenames = []
        for split, filename in SPLITS:
            if split in self.splits:
                split_filenames.append(filename)
        
        # Load annotations from each split file
        for filename in split_filenames:
            filepath = os.path.join(self.babel_dir, f'{filename}.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    result.update(data)
                if self.verbose:
                    print(f"Loaded {len(data)} samples from {filename}.json")
            else:
                if self.verbose:
                    print(f"Warning: {filepath} not found")
        
        return result
    
    def _extract_sequences(self) -> List[Dict[str, Any]]:
        """Extract all sequences from BABEL annotations."""
        sequences = []
        empty_annotations = []
        
        for sid, sample in self.raw_babel.items():
            # Get the relative path to the AMASS file
            amass_file_relative_path = "/".join(sample["feat_p"].split("/")[1:])
            
            annotations = []
            
            # Check for frame-level annotations first (if preferred)
            if self.prefer_frame_annotations and sample.get('frame_ann') is not None:
                for label in sample['frame_ann']['labels']:
                    annotations.append({
                        'act_cat': label['act_cat'],
                        'proc_label': label['proc_label'],
                        'raw_label': label['raw_label'],
                        'start_t': label['start_t'],
                        'end_t': label['end_t'],
                        'annotation_type': 'frame'
                    })
            # Fall back to sequence-level annotations
            elif sample.get('seq_ann') is not None:
                for label in sample['seq_ann']['labels']:
                    annotations.append({
                        'act_cat': label['act_cat'],
                        'proc_label': label['proc_label'],
                        'raw_label': label['raw_label'],
                        'start_t': 0,
                        'end_t': sample["dur"],
                        'annotation_type': 'sequence'
                    })
            # Try frame annotations if sequence not available and not preferred
            elif sample.get('frame_ann') is not None:
                for label in sample['frame_ann']['labels']:
                    annotations.append({
                        'act_cat': label['act_cat'],
                        'proc_label': label['proc_label'],
                        'raw_label': label['raw_label'],
                        'start_t': label['start_t'],
                        'end_t': label['end_t'],
                        'annotation_type': 'frame'
                    })
            else:
                empty_annotations.append(sid)
                continue
            
            # Sort annotations by start time
            annotations = sorted(annotations, key=lambda x: x['start_t'])
            
            # Add sequence to list
            sequences.append({
                'sid': sid,
                'amass_file_path': amass_file_relative_path,
                'annotations': annotations,
                'duration': sample.get("dur", annotations[-1]['end_t'] if annotations else 0)
            })
        
        if empty_annotations and self.verbose:
            print(f"Warning: {len(empty_annotations)} samples have no annotations: {empty_annotations[:10]}...")
        
        return sequences
    
    def _create_windows(self) -> List[Dict[str, Any]]:
        """Create windowed samples from sequences."""
        windows = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            if self.window_size == -1:
                # No windowing - use whole sequence
                for ann_idx, annotation in enumerate(sequence['annotations']):
                    duration = annotation['end_t'] - annotation['start_t']
                    if duration >= self.min_window_duration:
                        windows.append({
                            'sequence_idx': seq_idx,
                            'annotation_idx': ann_idx,
                            'start_time': annotation['start_t'],
                            'end_time': annotation['end_t'],
                            'duration': duration,
                            'label': annotation['proc_label'],
                            'act_cat': annotation['act_cat'],
                            'raw_label': annotation['raw_label'],
                            'annotation_type': annotation['annotation_type'],
                            'window_type': 'full_annotation'
                        })
            else:
                # Create sliding windows
                for annotation in sequence['annotations']:
                    start_frame = int(annotation['start_t'] * self.fps)
                    end_frame = int(annotation['end_t'] * self.fps)
                    annotation_length = end_frame - start_frame
                    
                    if annotation_length < self.window_size:
                        # Annotation is shorter than window size - skip or pad
                        if annotation_length >= self.window_size * 0.5:  # At least 50% of window size
                            windows.append({
                                'sequence_idx': seq_idx,
                                'annotation_idx': len(sequence['annotations']),  # Special idx for short annotations
                                'start_time': annotation['start_t'],
                                'end_time': annotation['end_t'],
                                'duration': annotation['end_t'] - annotation['start_t'],
                                'label': annotation['proc_label'],
                                'act_cat': annotation['act_cat'],
                                'raw_label': annotation['raw_label'],
                                'annotation_type': annotation['annotation_type'],
                                'window_type': 'short_annotation',
                                'start_frame': start_frame,
                                'end_frame': end_frame
                            })
                    else:
                        # Create sliding windows within this annotation
                        for window_start_frame in range(start_frame, end_frame - self.window_size + 1, self.window_stride):
                            window_end_frame = window_start_frame + self.window_size
                            window_start_time = window_start_frame / self.fps
                            window_end_time = window_end_frame / self.fps
                            
                            windows.append({
                                'sequence_idx': seq_idx,
                                'annotation_idx': len(sequence['annotations']),  # Special idx for windows
                                'start_time': window_start_time,
                                'end_time': window_end_time,
                                'duration': self.window_size / self.fps,
                                'label': annotation['proc_label'],
                                'act_cat': annotation['act_cat'],
                                'raw_label': annotation['raw_label'],
                                'annotation_type': annotation['annotation_type'],
                                'window_type': 'sliding_window',
                                'start_frame': window_start_frame,
                                'end_frame': window_end_frame,
                                'parent_annotation_start': annotation['start_t'],
                                'parent_annotation_end': annotation['end_t']
                            })
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a windowed sample."""
        window = self.windows[idx]
        sequence = self.sequences[window['sequence_idx']]
        
        # Load motion data using the provided motion loader
        motion_data = self.motion_loader(
            path=sequence['amass_file_path'],
            start=window['start_time'],
            end=window['end_time']
        )
        
        # Prepare output
        output = {
            'motion_data': motion_data,
            'label': window['label'],
            'act_cat': window['act_cat'],
            'raw_label': window['raw_label'],
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'duration': window['duration'],
            'sid': sequence['sid'],
            'window_type': window['window_type'],
            'annotation_type': window['annotation_type']
        }
        
        # Add additional info for windowed data
        if self.window_size != -1:
            output.update({
                'start_frame': window.get('start_frame'),
                'end_frame': window.get('end_frame'),
                'window_size': self.window_size,
                'window_stride': self.window_stride
            })
            
            if 'parent_annotation_start' in window:
                output.update({
                    'parent_annotation_start': window['parent_annotation_start'],
                    'parent_annotation_end': window['parent_annotation_end']
                })
        
        return output
    
    def get_label_statistics(self) -> Dict[str, int]:
        """Get statistics about label distribution."""
        label_counts = {}
        act_cat_counts = {}
        
        for window in self.windows:
            label = window['label']
            act_cat = window['act_cat']
            
            label_counts[label] = label_counts.get(label, 0) + 1
            act_cat_counts[act_cat] = act_cat_counts.get(act_cat, 0) + 1
        
        return {
            'labels': label_counts,
            'action_categories': act_cat_counts,
            'total_windows': len(self.windows)
        }
    
    def filter_by_labels(self, allowed_labels: List[str]) -> 'BabelWindowedDataset':
        """Create a new dataset filtered by specific labels."""
        filtered_windows = [
            window for window in self.windows 
            if window['label'] in allowed_labels
        ]
        
        # Create a copy of the dataset with filtered windows
        new_dataset = BabelWindowedDataset.__new__(BabelWindowedDataset)
        new_dataset.__dict__.update(self.__dict__)
        new_dataset.windows = filtered_windows
        
        return new_dataset
    
    def filter_by_action_categories(self, allowed_categories: List[str]) -> 'BabelWindowedDataset':
        """Create a new dataset filtered by specific action categories."""
        filtered_windows = [
            window for window in self.windows 
            if window['act_cat'] in allowed_categories
        ]
        
        # Create a copy of the dataset with filtered windows
        new_dataset = BabelWindowedDataset.__new__(BabelWindowedDataset)
        new_dataset.__dict__.update(self.__dict__)
        new_dataset.windows = filtered_windows
        
        return new_dataset
