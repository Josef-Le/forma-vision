"""Label schema for forma-vision.

One record per photo. Bulk format: a single labels.json / labels.yaml /
labels.jsonl next to an images/ directory:

    dataset/
      images/
        2026-09-01_front.jpg
        ...
      labels.yaml            (or labels.json / labels.jsonl)

Record fields (all optional except `image`; train on whatever labels exist —
missing targets are masked out of the loss):

    image: "2026-09-01_front.jpg"     # path relative to images/
    pose: "Front Relaxed"             # free text, used as a conditioning token
    bf: 14.5                          # measured body-fat percent (ground truth)
    bf_method: "DEXA"                 # DEXA | Caliper | BIA | Navy | judge | estimate
    muscles:                          # 1-10 scores, any subset of MUSCLES
      "Side delts": {size: 6, definition: 5}
    judge_score: 6.5                  # overall judge score 1-10
    weight_kg: 80.2
    measurements: {waist: 82, chest: 104, arm: 40, thigh: 60, hips: 96}
    sex: "male"
    height_cm: 178
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

MUSCLES = [
    "Chest", "Lats", "Upper back", "Traps", "Front delts", "Side delts",
    "Rear delts", "Biceps", "Triceps", "Forearms", "Abs", "Glutes",
    "Quads", "Hamstrings", "Calves",
]
MEASURE_SITES = ["waist", "chest", "arm", "thigh", "hips"]

# ground-truth quality → loss weight (a DEXA label teaches harder than a guess)
METHOD_WEIGHT = {
    "DEXA": 1.0, "Caliper": 0.8, "BIA": 0.55, "Navy": 0.6,
    "judge": 0.5, "estimate": 0.3, None: 0.3,
}


@dataclass
class Record:
    image: str
    pose: Optional[str] = None
    bf: Optional[float] = None
    bf_method: Optional[str] = None
    muscles: dict = field(default_factory=dict)
    judge_score: Optional[float] = None
    weight_kg: Optional[float] = None
    measurements: dict = field(default_factory=dict)
    sex: Optional[str] = None
    height_cm: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> list[str]:
        problems = []
        if self.bf is not None and not (2 <= float(self.bf) <= 60):
            problems.append(f"{self.image}: bf {self.bf} out of range 2-60")
        for name, ms in (self.muscles or {}).items():
            if name not in MUSCLES:
                problems.append(f"{self.image}: unknown muscle '{name}'")
            for k in ("size", "definition"):
                v = (ms or {}).get(k)
                if v is not None and not (1 <= float(v) <= 10):
                    problems.append(f"{self.image}: {name}.{k}={v} out of 1-10")
        return problems


def load_labels(path: Path) -> list[Record]:
    """Load labels.json / labels.yaml / labels.jsonl into Records."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # lazy: only needed for yaml datasets
        raw = yaml.safe_load(text)
    elif path.suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
    if isinstance(raw, dict):  # allow {"records": [...]}
        raw = raw.get("records", raw.get("labels", []))
    records = [Record.from_dict(d) for d in raw]
    problems = [p for r in records for p in r.validate()]
    if problems:
        raise ValueError("Label problems:\n" + "\n".join(problems[:50]))
    return records


def find_labels_file(root: Path) -> Path:
    for name in ("labels.yaml", "labels.yml", "labels.json", "labels.jsonl"):
        p = Path(root) / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No labels.(yaml|json|jsonl) found in {root}")
