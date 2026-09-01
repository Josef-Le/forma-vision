"""PyTorch dataset: images/ + labels file -> tensors with per-target masks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .labels import MUSCLES, METHOD_WEIGHT, Record, find_labels_file, load_labels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _split_of(name: str, val_frac: float) -> str:
    """Deterministic split by filename hash (stable across runs/machines)."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000
    return "val" if h < int(val_frac * 1000) else "train"


class PhysiqueDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train",
                 img_size: int = 384, val_frac: float = 0.15,
                 augment: bool = True):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        records = load_labels(find_labels_file(self.root))
        records = [r for r in records if (self.img_dir / r.image).exists()]
        if not records:
            raise ValueError(f"No records with existing images under {self.img_dir}")
        self.records: list[Record] = [
            r for r in records
            if split == "all" or _split_of(r.image, val_frac) == split
        ]
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.records)

    # -- targets ---------------------------------------------------------
    def _targets(self, r: Record):
        bf = torch.tensor([float(r.bf) if r.bf is not None else 0.0])
        bf_mask = torch.tensor([0.0 if r.bf is None else METHOD_WEIGHT.get(r.bf_method, 0.3)])

        size = torch.zeros(len(MUSCLES))
        deff = torch.zeros(len(MUSCLES))
        m_mask = torch.zeros(len(MUSCLES))
        for i, name in enumerate(MUSCLES):
            ms = (r.muscles or {}).get(name)
            if ms and ms.get("size") is not None:
                size[i] = float(ms["size"])
                deff[i] = float(ms.get("definition", ms["size"]))
                m_mask[i] = 1.0

        judge = torch.tensor([float(r.judge_score) if r.judge_score is not None else 0.0])
        judge_mask = torch.tensor([0.0 if r.judge_score is None else 1.0])
        return bf, bf_mask, size, deff, m_mask, judge, judge_mask

    # -- image -----------------------------------------------------------
    def _load_image(self, r: Record) -> torch.Tensor:
        img = Image.open(self.img_dir / r.image)
        img = ImageOps.exif_transpose(img).convert("RGB")
        s = self.img_size
        if self.augment:
            # conservative aug: physique photos must keep proportions honest —
            # no horizontal flip (left/right asymmetry is a *label*), no heavy crops.
            from torchvision import transforms as T
            tf = T.Compose([
                T.Resize(int(s * 1.08)),
                T.CenterCrop(s),
                T.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            from torchvision import transforms as T
            tf = T.Compose([
                T.Resize(int(s * 1.08)),
                T.CenterCrop(s),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        return tf(img)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        x = self._load_image(r)
        bf, bf_mask, size, deff, m_mask, judge, judge_mask = self._targets(r)
        return {
            "image": x,
            "bf": bf, "bf_mask": bf_mask,
            "size": size, "definition": deff, "muscle_mask": m_mask,
            "judge": judge, "judge_mask": judge_mask,
            "name": r.image,
        }

    def label_stats(self) -> dict:
        n = len(self.records)
        return {
            "records": n,
            "with_bf": sum(1 for r in self.records if r.bf is not None),
            "with_muscles": sum(1 for r in self.records if r.muscles),
            "with_judge": sum(1 for r in self.records if r.judge_score is not None),
        }
