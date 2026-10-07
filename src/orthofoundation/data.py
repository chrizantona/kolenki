"""Deterministic MRI sampling. No neighbouring slices disguised as RGB here."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]
N_SLOTS = 6
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def ids_hash(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def build_table(series, counts):
    """One series per slot: deepest cache first, UID as deterministic tie-break."""
    table = {}
    for row in series.itertuples():
        if row.Anatomical_Plane not in PLANES or int(row.Fat_Suppression) not in (0, 1):
            raise RuntimeError("Unexpected anatomical plane / fat suppression value")
        if counts.get(row.SeriesInstanceUID, 0) < 3:
            continue
        slot = PLANES.index(row.Anatomical_Plane) + 3 * (1 - int(row.Fat_Suppression))
        table.setdefault(row.StudyInstanceUID, [[] for _ in range(N_SLOTS)])[slot].append(row.SeriesInstanceUID)
    for slots in table.values():
        for choices in slots:
            choices.sort(key=lambda uid: (-counts[uid], uid))
    return table


def slice_indices(n, k):
    """Even deterministic centres, preserving genuinely short sequences."""
    if n < 3 or k < 1:
        raise ValueError("Need at least three available slices and one selected slice")
    return np.clip(np.rint(np.linspace(.04 * (n - 1), .96 * (n - 1), k)).astype(int), 0, n - 1)


class SliceStudyDataset(torch.utils.data.Dataset):
    def __init__(self, ids, table, loader, k):
        self.ids, self.table, self.loader, self.k = list(ids), table, loader, k

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uid = self.ids[index]
        x = np.zeros((N_SLOTS, self.k, 384, 384), dtype=np.uint8)
        mask = np.zeros(N_SLOTS, dtype=bool)
        positions = np.zeros((N_SLOTS, self.k), dtype=np.float32)
        selected = []
        for slot, choices in enumerate(self.table.get(uid, [[] for _ in range(N_SLOTS)])):
            if not choices:
                selected.append("")
                continue
            volume = self.loader(choices[0])
            centres = slice_indices(len(volume), self.k)
            x[slot] = volume[centres]
            mask[slot] = True
            positions[slot] = centres / (len(volume) - 1)
            selected.append(choices[0])
        if not mask.any():
            raise RuntimeError(f"No usable MRI series: {uid}")
        return uid, torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(positions), selected


def make_views(slices, resolution=224, fov=.92):
    """Single grayscale slice -> repeated RGB -> central crop/resize -> ImageNet norm.

    Input cache has canonical orientation, physical 153.6mm FOV, uint8 percentile
    intensity normalization. We retain .92 of that FOV (~141.3mm). This is a new
    preprocessing choice, not an exact reproduction of OrthoFoundation's data.
    """
    if slices.ndim != 3 or slices.dtype != torch.uint8:
        raise ValueError("Expected uint8 [N,H,W] MRI slices")
    if not 0 < fov <= 1 or resolution % 16:
        raise ValueError("FOV must be in (0,1], resolution divisible by DINOv3 patch size 16")
    side = min(slices.shape[-2:])
    crop = max(1, round(side * fov))
    top, left = (slices.shape[-2] - crop) // 2, (slices.shape[-1] - crop) // 2
    x = slices[:, None, top:top + crop, left:left + crop].float().div_(255)
    x = F.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False, antialias=True)
    x = x.expand(-1, 3, -1, -1)
    mean = x.new_tensor(MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def validated_labels(labels_path, data_dir):
    labels = pd.read_csv(labels_path, index_col=0)
    official = pd.read_csv(Path(data_dir) / "train.csv").set_index("StudyInstanceUID")
    if not labels.index.is_unique or not official.index.is_unique or set(labels.index) != set(official.index):
        raise RuntimeError("Label UID coverage differs from official training table")
    gold = official.index[official[LABELS].notna().all(axis=1)]
    if not labels.is_gold.isin([0, 1]).all() or set(labels.index[labels.is_gold.eq(1)]) != set(gold):
        raise RuntimeError("Gold exclusion IDs differ from official labels")
    targets = labels[LABELS].to_numpy(dtype=np.float32)
    if not np.isfinite(targets).all() or not ((targets >= 0) & (targets <= 1)).all():
        raise RuntimeError("Weak labels must be finite probabilities in [0,1]")
    if not labels.loc[labels.is_gold.eq(0), "fold"].isin(range(5)).all():
        raise RuntimeError("Weak study folds must be 0..4")
    return labels, official, list(gold)
