"""Run a trained checkpoint on one image or a folder.

    python -m formavision.infer --ckpt runs/xxx/best.pt --input photo.jpg
Outputs JSON matching the Forma app's scan schema (bf + per-muscle scores).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .labels import MUSCLES
from .model import FormaVision


def load_model(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    backbone = ck.get("args", {}).get("backbone", "convnext_tiny")
    model = FormaVision(backbone, pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    return model, ck.get("args", {})


@torch.no_grad()
def predict(model, path: Path, img_size: int, device: str) -> dict:
    from torchvision import transforms as T
    tf = T.Compose([
        T.Resize(int(img_size * 1.08)), T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    out = model(x)
    return {
        "image": path.name,
        "bf": round(out["bf"].item(), 1),
        "judge_score": round(out["judge"].item(), 1),
        "muscles": [
            {"name": m,
             "size": round(out["size"][0, i].item(), 1),
             "definition": round(out["definition"][0, i].item(), 1)}
            for i, m in enumerate(MUSCLES)
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--img-size", type=int, default=None)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train_args = load_model(args.ckpt, device)
    size = args.img_size or train_args.get("img_size", 384)

    inp = Path(args.input)
    paths = sorted(p for p in ([inp] if inp.is_file() else inp.iterdir())
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    results = [predict(model, p, size, device) for p in paths]
    print(json.dumps(results if len(results) > 1 else results[0], indent=1))


if __name__ == "__main__":
    main()
