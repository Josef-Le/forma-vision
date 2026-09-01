"""Vision backbone + multi-head regression for physique assessment.

Backbone: any timm model (default convnext_tiny — strong at 384px, exportable).
Heads:
  bf         -> 1   (body-fat %)
  size       -> 15  (muscle size scores, 1-10)
  definition -> 15  (muscle definition scores, 1-10)
  judge      -> 1   (overall judge score, 1-10)
Losses are masked per-sample/per-target so partially labeled data still trains.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .labels import MUSCLES


class FormaVision(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0)  # feature extractor
        feat = self.backbone.num_features
        self.neck = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat, 512),
            nn.GELU(),
        )
        n = len(MUSCLES)
        self.head_bf = nn.Linear(512, 1)
        self.head_size = nn.Linear(512, n)
        self.head_def = nn.Linear(512, n)
        self.head_judge = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> dict:
        f = self.neck(self.backbone(x))
        return {
            # scaled sigmoids keep outputs in physically valid ranges
            "bf": torch.sigmoid(self.head_bf(f)) * 58.0 + 2.0,        # 2..60 %
            "size": torch.sigmoid(self.head_size(f)) * 9.0 + 1.0,     # 1..10
            "definition": torch.sigmoid(self.head_def(f)) * 9.0 + 1.0,
            "judge": torch.sigmoid(self.head_judge(f)) * 9.0 + 1.0,
        }


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """SmoothL1 where mask==0 rows/elements contribute nothing.
    mask may carry per-label weights (e.g. DEXA=1.0 vs BIA=0.55)."""
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    weighted = loss * mask
    denom = mask.sum().clamp(min=1e-6)
    return weighted.sum() / denom


def total_loss(out: dict, batch: dict, w_bf: float = 1.0, w_muscle: float = 1.0,
               w_judge: float = 0.5) -> tuple[torch.Tensor, dict]:
    l_bf = masked_smooth_l1(out["bf"], batch["bf"], batch["bf_mask"], beta=2.0)
    l_sz = masked_smooth_l1(out["size"], batch["size"], batch["muscle_mask"])
    l_df = masked_smooth_l1(out["definition"], batch["definition"], batch["muscle_mask"])
    l_j = masked_smooth_l1(out["judge"], batch["judge"], batch["judge_mask"])
    total = w_bf * l_bf + w_muscle * (l_sz + l_df) + w_judge * l_j
    return total, {"bf": l_bf.item(), "size": l_sz.item(),
                   "definition": l_df.item(), "judge": l_j.item()}
