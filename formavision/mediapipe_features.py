"""Extract MediaPipe pose keypoints + geometric physique features per image.

    python -m formavision.mediapipe_features --data ./dataset --out features.csv

These are the same distance-invariant ratios Forma Lab computes in the browser
(shoulder/hip ratio, limb symmetry, tilts), produced in bulk for a dataset.
Use them three ways:
  1. sanity-check labels (a "front relaxed" photo with a 40° shoulder tilt is
     probably mis-posed — filter it or fix the pose tag);
  2. train a cheap tabular baseline (features -> bf) to compare the CNN against;
  3. late-fusion: concatenate to the backbone features for a small accuracy bump.

Requires: pip install mediapipe  (downloads the pose model on first run).
"""
from __future__ import annotations

import argparse
import csv
import math
import urllib.request
from pathlib import Path

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task")

# MediaPipe pose landmark indices
IDX = dict(nose=0, shL=11, shR=12, elL=13, elR=14, wrL=15, wrR=16,
           hipL=23, hipR=24, knL=25, knR=26, anL=27, anR=28)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def features_from_landmarks(pts: dict) -> dict:
    """pts: name -> (x, y) in pixels. Mirrors forma-lab.html analyze()."""
    mid = lambda a, b: ((pts[a][0] + pts[b][0]) / 2, (pts[a][1] + pts[b][1]) / 2)
    mid_sh, mid_hip, mid_an = mid("shL", "shR"), mid("hipL", "hipR"), mid("anL", "anR")
    shoulder_w = _dist(pts["shL"], pts["shR"])
    hip_w = _dist(pts["hipL"], pts["hipR"])
    arm_l = _dist(pts["shL"], pts["elL"]) + _dist(pts["elL"], pts["wrL"])
    arm_r = _dist(pts["shR"], pts["elR"]) + _dist(pts["elR"], pts["wrR"])
    leg_l = _dist(pts["hipL"], pts["knL"]) + _dist(pts["knL"], pts["anL"])
    leg_r = _dist(pts["hipR"], pts["knR"]) + _dist(pts["knR"], pts["anR"])
    torso = _dist(mid_sh, mid_hip)
    sym = lambda l, r: min(l, r) / max(l, r) * 100 if max(l, r) > 0 else 0.0
    tilt = lambda a, b: math.degrees(math.atan2(pts[b][1] - pts[a][1],
                                                pts[b][0] - pts[a][0] + 1e-6))
    return {
        "shoulder_hip_ratio": round(shoulder_w / hip_w, 3) if hip_w else 0,
        "torso_leg_ratio": round(torso / ((leg_l + leg_r) / 2), 3) if leg_l + leg_r else 0,
        "arm_symmetry_pct": round(sym(arm_l, arm_r), 1),
        "leg_symmetry_pct": round(sym(leg_l, leg_r), 1),
        "shoulder_tilt_deg": round(tilt("shL", "shR"), 1),
        "hip_tilt_deg": round(tilt("hipL", "hipR"), 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir with images/")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--model", default="pose_landmarker_full.task")
    args = ap.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"downloading pose model -> {model_path}")
        urllib.request.urlretrieve(MODEL_URL, model_path)

    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE, num_poses=1))

    img_dir = Path(args.data) / "images"
    rows = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        image = mp.Image.create_from_file(str(p))
        res = landmarker.detect(image)
        if not res.pose_landmarks:
            rows.append({"image": p.name, "detected": 0})
            continue
        lms = res.pose_landmarks[0]
        pts = {k: (lms[i].x * image.width, lms[i].y * image.height)
               for k, i in IDX.items()}
        rows.append({"image": p.name, "detected": 1,
                     **features_from_landmarks(pts)})

    fieldnames = ["image", "detected", "shoulder_hip_ratio", "torso_leg_ratio",
                  "arm_symmetry_pct", "leg_symmetry_pct",
                  "shoulder_tilt_deg", "hip_tilt_deg"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    det = sum(r.get("detected", 0) for r in rows)
    print(f"{det}/{len(rows)} images had a detected pose -> {args.out}")


if __name__ == "__main__":
    main()
