import torch

from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import default_collate


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_x_dict(lst_x_dict: List, *, device: Optional[str] = None) -> Dict:
    x = collate_tensor_with_padding([x_dict["x"] for x_dict in lst_x_dict])
    if device is not None:
        x = x.to(device)
    length = [x_dict["length"] for x_dict in lst_x_dict]
    mask = length_to_mask(length, device=x.device)
    batch = {"x": x, "length": length, "mask": mask}
    return batch


def collate_text_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    x_dict_keys = [key for key in keys if "x_dict" in key]
    other_keys = [key for key in keys if "x_dict" not in key]

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    for key in x_dict_keys:
        batch[key] = collate_x_dict([x[key] for x in lst_elements], device=device)
    return batch

def segments_collate_text_motion(batch):
    motion_seqs = [item["motion_x_dict"]["x"] for item in batch]
    lengths = [seq.shape[0] for seq in motion_seqs]
    max_len = max(lengths)
    feat_dim = motion_seqs[0].shape[1]

    # Padding motion sequences
    padded_motion = torch.zeros((len(batch), max_len, feat_dim), dtype=motion_seqs[0].dtype)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, (seq, length) in enumerate(zip(motion_seqs, lengths)):
        padded_motion[i, :length] = seq
        mask[i, :length] = True

    # Collate transition masks
    transition_masks = [item["transition_mask"] for item in batch]
    padded_transition = torch.zeros((len(batch), max_len), dtype=transition_masks[0].dtype)
    for i, (mask_i, length) in enumerate(zip(transition_masks, lengths)):
        padded_transition[i, :length] = mask_i

    # Sentences (list of lists)
    texts = [item["text"] for item in batch]
    sent_embs = [item["sent_emb"] for item in batch]
    keyids = [item["keyid"] for item in batch]

    return {
        "motion_x_dict": {
            "x": padded_motion,
            "length": lengths,
            "mask": mask
        },
        "transition_mask": padded_transition,
        "text": texts,
        "sent_emb": sent_embs,
        "keyid": keyids
    }