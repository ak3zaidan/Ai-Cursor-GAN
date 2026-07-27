
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import resample_sequence, per_sequence_normalize

class CursorDataset(Dataset):
    def __init__(self, json_path: str, fixed_length: int = 128, use_time_deltas: bool = False, per_sequence_norm: bool = True):
        with open(json_path, "r") as f:
            payload = json.load(f)
        self.seqs = payload["data"]
        self.fixed_length = fixed_length
        self.use_time_deltas = use_time_deltas
        self.per_sequence_norm = per_sequence_norm

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        resampled = resample_sequence(seq, self.fixed_length)  # [T, 3]
        xy = resampled[:, :2]
        t = resampled[:, 2]

        if self.per_sequence_norm:
            xy, mean, std = per_sequence_normalize(xy)
        else:
            mean = np.zeros((1,2))
            std = np.ones((1,2))

        if self.use_time_deltas:
            dt = np.diff(t, prepend=t[0])
            dt = (dt - dt.mean()) / (dt.std() + 1e-8)
            feats = np.concatenate([xy, dt[:, None]], axis=1)  # [T, 3]
        else:
            feats = xy  # [T, 2]

        feats = feats.T.astype(np.float32)  # [C, T]
        meta = {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}
        return torch.from_numpy(feats), meta
