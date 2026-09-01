"""Distillation labeler: run the LLM judge over a whole dataset to generate
pseudo-labels, so the CV model trains at scale BEFORE any ground truth exists.

    export ANTHROPIC_API_KEY=sk-ant-...
    python -m formavision.pseudo_label --data ./dataset            # label everything unlabeled
    python -m formavision.pseudo_label --data ./dataset --relabel  # redo all

Each image gets: bf estimate (bf_method="estimate" → trains at 0.3x weight vs
DEXA's 1.0x), the 15 muscle size/definition scores, and a judge score — the
same rubric the app's live judge uses, so the distilled CNN reproduces it.
Cost control: images are downscaled before sending; --limit caps a run; already
labeled records are skipped unless --relabel. Requires: pip install anthropic
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

from .labels import MUSCLES, find_labels_file, load_labels  # noqa: F401 (schema ref)

PROMPT = f"""You are an experienced bodybuilding judge and physique coach assessing ONE photo for a training-data labeling task. Be honest and consistent, never flattering.
Calibration: 5 = average trained recreational lifter; most recreational lifters land 3.5-6.0 overall; reserve 8+ for competitive stage conditioning.
Estimate body-fat percent from visual cues (ab visibility, vascularity, oblique/lower-back folds).
Score EVERY one of these muscles for size and definition, 1-10: {", ".join(MUSCLES)}.
If no clearly visible person, or the image is not a physique photo, reply with exactly {{"skip": true}}.
Reply with ONLY this JSON:
{{"bf": number, "judge_score": number,
 "muscles": {{"<muscle name>": {{"size": number, "definition": number}}, ...}} }}
Include all {len(MUSCLES)} muscles."""


def _shrink(path: Path, max_px: int = 720) -> tuple[str, str]:
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _parse(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    data = json.loads(text[start:end + 1])
    if data.get("skip"):
        return None
    return data


def label_one(client, model: str, img_path: Path) -> dict | None:
    b64, mime = _shrink(img_path)
    msg = client.messages.create(
        model=model, max_tokens=1500,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            {"type": "text", "text": PROMPT},
        ]}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    data = _parse(text)
    if data is None:
        return None
    return _to_record(data)


def label_one_openrouter(api_key: str, model: str, img_path: Path) -> dict | None:
    """Same labeling call through OpenRouter's OpenAI-compatible API (stdlib only)."""
    import urllib.request
    b64, mime = _shrink(img_path)
    body = json.dumps({
        "model": model, "max_tokens": 1500,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com", "X-Title": "forma-vision"})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    if resp.get("error"):
        raise RuntimeError(str(resp["error"])[:200])
    text = resp["choices"][0]["message"]["content"] or ""
    data = _parse(text)
    return None if data is None else _to_record(data)


def _to_record(data: dict) -> dict | None:
    out = {}
    if isinstance(data.get("bf"), (int, float)) and 2 <= data["bf"] <= 60:
        out["bf"] = round(float(data["bf"]), 1)
        out["bf_method"] = "estimate"          # trains at reduced weight
    if isinstance(data.get("judge_score"), (int, float)):
        out["judge_score"] = round(min(10, max(1, float(data["judge_score"]))), 1)
    muscles = {}
    for name, ms in (data.get("muscles") or {}).items():
        if name in MUSCLES and isinstance(ms, dict):
            try:
                muscles[name] = {"size": round(min(10, max(1, float(ms["size"]))), 1),
                                 "definition": round(min(10, max(1, float(ms.get("definition", ms["size"])))), 1)}
            except (KeyError, TypeError, ValueError):
                continue
    if muscles:
        out["muscles"] = muscles
    return out or None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir (images/ + labels file)")
    ap.add_argument("--model", default="auto", help="vision model for labeling (auto = provider default)")
    ap.add_argument("--limit", type=int, default=0, help="max images this run (0 = all)")
    ap.add_argument("--relabel", action="store_true", help="also redo already-labeled records")
    args = ap.parse_args(argv)

    import os
    or_key = os.environ.get("OPENROUTER_API_KEY")
    an_key = os.environ.get("ANTHROPIC_API_KEY")
    if or_key:
        model = args.model if args.model != "auto" else "anthropic/claude-sonnet-4.5"
        label_fn = lambda p: label_one_openrouter(or_key, model, p)  # noqa: E731
        provider = "openrouter"
    elif an_key:
        try:
            import anthropic
        except ImportError:
            raise SystemExit("pip install anthropic")
        client = anthropic.Anthropic()
        model = args.model if args.model != "auto" else "claude-sonnet-4-5"
        label_fn = lambda p: label_one(client, model, p)  # noqa: E731
        provider = "anthropic"
    else:
        raise SystemExit("set OPENROUTER_API_KEY or ANTHROPIC_API_KEY")

    root = Path(args.data)
    labels_path = find_labels_file(root) if any(
        (root / n).exists() for n in ("labels.json", "labels.yaml", "labels.yml", "labels.jsonl")
    ) else root / "labels.json"
    records = json.loads(labels_path.read_text()) if labels_path.suffix == ".json" and labels_path.exists() else \
        [r.__dict__ if hasattr(r, "__dict__") else r for r in ([] if not labels_path.exists() else load_labels(labels_path))]
    if records and hasattr(records[0], "image"):
        records = [{k: v for k, v in r.__dict__.items() if v not in (None, {}, [])} for r in records]
    by_image = {r["image"]: r for r in records}

    # every image on disk gets a record
    for p in sorted((root / "images").iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and p.name not in by_image:
            rec = {"image": p.name}
            records.append(rec)
            by_image[p.name] = rec

    todo = [r for r in records if args.relabel or
            (r.get("bf") is None and not r.get("muscles"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(records)} records, labeling {len(todo)} via {provider} ({model})")

    done = 0
    for r in todo:
        img_path = root / "images" / r["image"]
        if not img_path.exists():
            continue
        try:
            lab = label_fn(img_path)
        except Exception as e:  # noqa: BLE001 — keep the batch alive
            print(f"  {r['image']}: {e}")
            time.sleep(3)
            continue
        if lab is None:
            print(f"  {r['image']}: skipped (no usable physique)")
            continue
        # never overwrite real ground truth with a pseudo-label
        if r.get("bf_method") not in (None, "estimate"):
            lab.pop("bf", None)
            lab.pop("bf_method", None)
        r.update(lab)
        done += 1
        if done % 10 == 0:
            (root / "labels.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"  …{done}/{len(todo)} (checkpoint saved)")
        time.sleep(0.4)

    (root / "labels.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"labeled {done} images -> {root/'labels.json'} — ready for: python -m formavision.train --data {root}")


if __name__ == "__main__":
    main()
