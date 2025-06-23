import os
import enum
import json
import torch

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

# TODO: add something to let the user decide whether to consider sequence annotations or frame annotations first or both at the same time
# frame first will first check if frames are densly annotated and add those to the dataset ignoring the sequence annotations
# sequence first will first check if sequences are densly annotated and add those to the dataset ignoring the frame annotations
# both will add all the sequences and frames to the dataset making samples sort of "redundent"

class SubSegmentsBabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotations_dir: str,
        splits: BabelDatasetSplit = BabelDatasetSplit.ALL,
        transform=None,
    ):
        self.splits = splits
        self.annotations_dir = annotations_dir
        self.transform = transform
        
        self.splits_filesnames = self.__process_splits()
      
        self.raw_babel = self.__process_babel()
                
        self.data, empty_annotations = self.__process_data()
        
        if len(empty_annotations) > 0:
            print(f"[warnning]: The following samples have no annotations({len(empty_annotations)}): {empty_annotations}")
                    
    def __process_splits(self):
        result = []
        
        for split, filename in SPLITS:
            if split in self.splits:
                result.append(filename)
                
        return result
    
    def __process_babel(self):
        result = {}
        
        for filename in self.splits_filesnames:
            with open(os.path.join(self.annotations_dir, f'{filename}.json'), "r") as file:
                data = json.load(file)
                result.update(data)
                
        return result
    
    def __process_data(self):
        sequences = []
        empty_annotations = []
        
        for sid in self.raw_babel:
            sample = self.raw_babel[sid]
            
            amass_file_relative_path = "/".join(sample["feat_p"].split("/")[1:])
            
            if sample.get('frame_ann') is not None:
                for sub_sequence in sample['frame_ann']['labels']:
                    sequences.append({
                        'sid': sid,
                        'amass_file_relative_path': amass_file_relative_path,
                        'act_cat': sub_sequence['act_cat'],
                        'proc_label': sub_sequence['proc_label'],
                        'raw_label': sub_sequence['raw_label'],
                        'start_t': sub_sequence['start_t'],
                        'end_t': sub_sequence['end_t'],
                    })
                continue
            elif sample.get('seq_ann') is not None:
                for sequence in sample["seq_ann"]["labels"]:
                    sequences.append({
                        'sid': sid,
                        'amass_file_relative_path': amass_file_relative_path,
                        'act_cat': sequence['act_cat'],
                        'proc_label': sequence['proc_label'],
                        'raw_label': sequence['raw_label'],
                        'start_t': 0,
                        # TODO: check if this coincide with the amass data
                        'end_t': sample["dur"],
                    })
                continue
            else:
                empty_annotations.append(sid)
                # raise ValueError(f"Neither frame annotations nor sequence annotations found for sample {sid}.")
            
        return sequences, empty_annotations
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class SegmentsBabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotations_dir: str,
        splits: BabelDatasetSplit = BabelDatasetSplit.ALL,
        transform=None,
    ):
        self.splits = splits
        self.annotations_dir = annotations_dir
        self.transform = transform
        
        self.splits_filesnames = self.__process_splits()
        self.raw_babel = self.__process_babel()
        self.data, empty_annotations = self.__process_data()
        
        if len(empty_annotations) > 0:
            print(f"[warning]: The following samples have no annotations ({len(empty_annotations)}): {empty_annotations}")
    
    def __process_splits(self):
        return [filename for split, filename in SPLITS if split in self.splits]
    
    def __process_babel(self):
        result = {}
        for filename in self.splits_filesnames:
            with open(os.path.join(self.annotations_dir, f'{filename}.json'), "r") as file:
                data = json.load(file)
                result.update(data)
        return result
    
    def __process_data(self):
        data = []
        empty = []
        
        for sid, sample in self.raw_babel.items():
            amass_file_relative_path = "/".join(sample["feat_p"].split("/")[1:])
            
            annotations = []
            if sample.get("frame_ann") is not None:
                for label in sample["frame_ann"]["labels"]:
                    annotations.append({
                        "act_cat": label["act_cat"],
                        "annotation": label["proc_label"],
                        "start_t": label["start_t"],
                        "end_t": label["end_t"]
                    })
            elif sample.get("seq_ann") is not None:
                for label in sample["seq_ann"]["labels"]:
                    annotations.append({
                        "act_cat": label["act_cat"],
                        "annotation": label["proc_label"],
                        "start_t": 0,
                        "end_t": sample["dur"]
                    })
            else:
                empty.append(sid)
                continue
            
            annotations = sorted(annotations, key=lambda annotation: annotation["start_t"])
            
            data.append({
                "sid": sid,
                "amass_file_relative_path": amass_file_relative_path,
                "annotations": annotations
            })
        
        return data, empty
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        output = self.data[idx]
        
        if self.transform:
            output = self.transform(output)
            
        return output
        
        
def make_annotations_continuous(sample):
    """
    Transform function that ensures annotations are continuous with no gaps.
    The end time of each annotation becomes the start time of the next annotation.
    
    Args:
        sample: Dictionary containing 'annotations' list and other sample data
        
    Returns:
        Modified sample with continuous annotations
    """
    if 'annotations' not in sample or len(sample['annotations']) <= 1:
        return sample
    
    # Create a copy to avoid modifying the original
    sample_copy = sample.copy()
    annotations = sample['annotations'].copy()
    
    # Sort annotations by start time to ensure proper ordering
    annotations = sorted(annotations, key=lambda x: x['start_t'])
    
    # Make annotations continuous
    for i in range(len(annotations) - 1):
        current_annotation = annotations[i]
        next_annotation = annotations[i + 1]
        
        # Set the end time of current annotation to start time of next annotation
        # This eliminates any gaps between annotations
        if current_annotation['end_t'] < next_annotation['start_t']:
            # If there's a gap, extend current annotation to fill it
            current_annotation['end_t'] = next_annotation['start_t']
        elif current_annotation['end_t'] > next_annotation['start_t']:
            # If there's an overlap, adjust the next annotation's start time
            next_annotation['start_t'] = current_annotation['end_t']
    
    sample_copy['annotations'] = annotations
    return sample_copy

def make_annotations_continuous_with_interpolation(sample):
    if 'annotations' not in sample or len(sample['annotations']) <= 1:
        return sample
    
    sample_copy = sample.copy()
    annotations = sample['annotations'].copy()
    
    annotations = sorted(annotations, key=lambda x: x['start_t'])
    
    for i in range(len(annotations) - 1):
        current_annotation = annotations[i]
        next_annotation = annotations[i + 1]
        
        # NOTE: there is a gap, we split it equally
        if current_annotation['end_t'] < next_annotation['start_t']:
            gap_midpoint = (current_annotation['end_t'] + next_annotation['start_t']) / 2
            current_annotation['end_t'] = gap_midpoint
            next_annotation['start_t'] = gap_midpoint
        # NOTE: there is an overlap
        elif current_annotation['end_t'] > next_annotation['start_t']:
            overlap_midpoint = (current_annotation['end_t'] + next_annotation['start_t']) / 2
            current_annotation['end_t'] = overlap_midpoint
            next_annotation['start_t'] = overlap_midpoint
    
    sample_copy['annotations'] = annotations
    
    return sample_copy

import copy

def make_annotations_continuous_extend_forward(
    sample,
    verbose=False,
    gap_threshold=1.0,
    overlap_threshold=1.0,
    epsilon=1e-6,
):
    if 'annotations' not in sample or len(sample['annotations']) <= 1:
        return sample, False, False
    
    sample_copy = copy.deepcopy(sample)
    annotations = sample_copy['annotations']
    annotations = sorted(annotations, key=lambda x: x['start_t'])
    
    gap = False
    overlap = False
    
    for i in range(len(annotations) - 1):
        current_annotation = annotations[i]
        next_annotation = annotations[i + 1]
        
        time_diff = next_annotation['start_t'] - current_annotation['end_t']

        if time_diff > gap_threshold + epsilon:
            gap = True
            if verbose:
                print(f"[warning]: gap of {time_diff:.2f}s between annotations {i} and {i + 1}.")
        elif time_diff < -overlap_threshold + epsilon:
            overlap = True
            if verbose:
                print(f"[warning]: overlap of {-time_diff:.2f}s between annotations {i} and {i + 1}.")
        
        current_annotation['end_t'] = next_annotation['start_t']
    
    sample_copy['annotations'] = annotations
    
    return sample_copy, overlap, gap