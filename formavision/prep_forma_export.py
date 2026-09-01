"""Turn a Forma app export (JSON) into a training dataset folder.

    python -m formavision.prep_forma_export --export forma-export.json --out ./dataset

Pairs every photo with:
  - measured body fat from the nearest ground-truth entry within --bf-window days
    (falls back to the AI scan estimate, weighted lower via bf_method="estimate")
  - the muscle scores of the AI scan that used the photo (or nearest same-day scan)
  - the per-photo judge score if present
  - nearest weight and tape measurements
"""
from __future__ import annotations

import argparse
import base64
import json
from datetime import date
from pathlib import Path


def _d(iso: str) -> date:
    return date.fromisoformat(iso[:10])


def _nearest(entries, target_iso, key="dateISO", window=None):
    if not entries:
        return None
    t = _d(target_iso)
    best, bd = None, None
    for e in entries:
        try:
            d = abs((_d(e.get(key) or e.get("id")) - t).days)
        except (ValueError, TypeError):
            continue
        if bd is None or d < bd:
            best, bd = e, d
    if best is not None and window is not None and bd > window:
        return None
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="Forma export JSON")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--bf-window", type=int, default=10,
                    help="max days between photo and ground-truth bf measurement")
    args = ap.parse_args(argv)

    data = json.loads(Path(args.export).read_text(encoding="utf-8"))
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # Lab "training bundle" (app → Lab → Export bundle): labels come pre-paired
    if data.get("type") == "forma-dataset":
        profile = data.get("profile") or {}
        records = []
        for i, r in enumerate(data.get("records", [])):
            img = r.get("img", "")
            if not img.startswith("data:image"):
                continue
            ext = "jpg" if "jpeg" in img[:30] else "png"
            name = f"{r.get('source','x')}_{i:04d}.{ext}"
            (out / "images" / name).write_bytes(base64.b64decode(img.split(",", 1)[1]))
            rec = {"image": name}
            for src_k, dst_k in (("pose", "pose"), ("bf", "bf"), ("bf_method", "bf_method"),
                                 ("judge_score", "judge_score"), ("muscles", "muscles")):
                if r.get(src_k) is not None:
                    rec[dst_k] = r[src_k]
            if profile.get("sex"):
                rec["sex"] = profile["sex"]
            if profile.get("height_cm"):
                rec["height_cm"] = profile["height_cm"]
            records.append(rec)
        (out / "labels.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
        n_gt = sum(1 for r in records if r.get("bf_method") not in (None, "estimate"))
        print(f"bundle: wrote {len(records)} records ({n_gt} with real bf ground truth) -> {out}")
        return

    photos = data.get("photos", [])
    scans = data.get("scans", [])
    truth = data.get("truth", [])
    weight = data.get("weight", [])
    measure = data.get("measure", [])
    profile = data.get("profile") or {}

    records = []
    for p in photos:
        img = p.get("img", "")
        if not img.startswith("data:image"):
            continue
        ext = "jpg" if "jpeg" in img[:30] else "png"
        name = f"{p.get('dateISO','unknown')}_{p.get('id','x')}.{ext}"
        (out / "images" / name).write_bytes(base64.b64decode(img.split(",", 1)[1]))

        rec = {"image": name, "pose": p.get("pose")}

        gt = _nearest(truth, p["dateISO"], window=args.bf_window)
        scan = next((s for s in scans if p.get("id") in (s.get("photoIds") or [])), None) \
            or _nearest(scans, p["dateISO"], window=3)
        if gt:
            rec["bf"] = gt["bf"]
            rec["bf_method"] = gt.get("method", "Caliper").split()[0]
        elif scan and scan.get("bf"):
            rec["bf"] = scan["bf"]
            rec["bf_method"] = "estimate"
        if scan:
            rec["muscles"] = {m["name"]: {"size": m["size"], "definition": m["definition"]}
                              for m in scan.get("muscles", [])}
        if p.get("judge"):
            rec["judge_score"] = p["judge"].get("score")
        w = _nearest(weight, p["dateISO"], key="id", window=7)
        if w:
            rec["weight_kg"] = w.get("kg")
        ms = _nearest(measure, p["dateISO"], key="id", window=21)
        if ms:
            rec["measurements"] = {k: v for k, v in ms.items()
                                   if k in ("waist", "chest", "arm", "thigh", "hips")}
        if profile.get("sex"):
            rec["sex"] = profile["sex"]
        if profile.get("height"):
            rec["height_cm"] = profile["height"]
        records.append(rec)

    (out / "labels.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    n_gt = sum(1 for r in records if r.get("bf_method") not in (None, "estimate"))
    print(f"wrote {len(records)} records ({n_gt} with real bf ground truth) -> {out}")


if __name__ == "__main__":
    main()
