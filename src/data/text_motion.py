import os
import json
import orjson

import torch
import numpy as np
import codecs as cs

from tqdm import tqdm
from torch.utils.data import Dataset

from .collate import collate_text_motion, segments_collate_text_motion

def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list

def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())

class TextMotionDataset(Dataset):
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
        tiny: bool = False,
        mode: str = "classic",
        text_only: bool = False
    ):
        if tiny:
            split = split + "_tiny"

        self.split = split
        self.keyids = read_split(path, split)
        self.mode = mode
        self.load_keyid, self.collate_fn = self._get_mode_attributes(self.mode)

        self.text_to_sent_emb = text_to_sent_emb
        self.text_to_token_emb = text_to_token_emb
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = split == "train"
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats
        self.failed_indices = []

        if preload:
            for index in tqdm(range(len(self)), desc="[preloading-dataset]"):
                try:
                    self[index]
                except Exception as exception:
                    self.failed_indices.append(index)
                    print("[text_motion.py] error preloading {index}".format(index=index))
                    print(exception)
                continue
            
            self.keyids = [self.keyids[i] for i in range(len(self.keyids)) if int(self.keyids[i]) not in self.failed_indices]

    def _get_mode_attributes(self, mode):
        if mode == "classic":
            return (self.load_keyid_classic, collate_text_motion)
        elif mode == "segmentation":
            return (self.load_keyid_segments, segments_collate_text_motion)
        elif mode == "classifier":
            # NOTE: default
            # return (self.load_keyid_classifier, collate_text_motion)
            # NOTE: for classifier evaluation
            return (self.load_keyid_classic, collate_text_motion)
        else:
            raise Exception("Invalid mode provided")

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        
        return self.load_keyid(keyid)

    def load_keyid_classic(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][index]

        text = annotation["text"]
        text_x_dict = self.text_to_token_emb(text)
        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
        )
        sent_emb = self.text_to_sent_emb(text)

        output = {
            "motion_x_dict": motion_x_dict,
            "text_x_dict": text_x_dict,
            "text": text,
            "keyid": keyid,
            "sent_emb": sent_emb,
        }
        return output
    
    def load_keyid_classifier(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][index]

        text = annotation["text"]

        output = {
            "text": text,
            "keyid": keyid,
            "length": annotation["end"] - annotation["start"],
        }
        return output
    
    def load_keyid_segments(self, keyid):
        ann = self.annotations[keyid]
        annotations = ann["annotations"]

        # NOTE: compute the motion span
        starts = [annotation["start"] for annotation in annotations]
        ends = [annotation["end"] for annotation in annotations]
        full_start = min(starts)
        full_end = max(ends)

        # NOTE: load the corresponding motion
        motion_x_dict = self.motion_loader(
            path=ann["path"],
            start=full_start,
            end=full_end,
        )
        
        num_frames = motion_x_dict["x"].shape[0]

        fps = 20

        transition_mask = torch.zeros(num_frames, dtype=torch.float)
        for annotation in annotations:
            if "transition" in annotation["text"].lower():
                rel_start = max(int((annotation["start"] - full_start) * fps), 0)
                rel_end = min(int((annotation["end"] - full_start) * fps), num_frames)
                transition_mask[rel_start:rel_end] = 1

        texts = [aannotation["text"] for aannotation in annotations]
        sent_embs = [self.text_to_sent_emb(text) for text in texts]

        output = {
            "motion_x_dict": motion_x_dict,
            "text": texts,
            "keyid": keyid,
            "sent_emb": sent_embs,
            "transition_mask": transition_mask,
        }

        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
