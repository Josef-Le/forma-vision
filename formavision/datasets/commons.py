"""Harvest genuinely free-licensed physique/posing/exercise images from
Wikimedia Commons, with a per-image attribution manifest.

    python -m formavision.datasets.commons --out dataset_commons --limit 40
    python -m formavision.datasets.commons --query "front double biceps" --limit 60

Only images whose license is in ALLOWED_LICENSES are kept (CC0, public domain,
CC BY, CC BY-SA). attribution.csv records author, license and source URL for
every file — publish it wherever the images are used. "Publicly visible" does
NOT mean freely licensed; this module keeps you on the right side of that line
automatically by reading each file's license metadata from the Commons API.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "forma-vision/1.0 (dataset builder; contact: repo owner)"}

ALLOWED_LICENSES = re.compile(
    r"^(cc0(?:[- ]\d\.\d)?|public domain|pd|cc[- ]by(?:[- ]sa)?(?:[- ]\d\.\d)?)$", re.I)

# default query set: one per mandatory pose + common exercises
DEFAULT_QUERIES = [
    "bodybuilding front double biceps", "bodybuilding lat spread",
    "bodybuilding side chest pose", "bodybuilding back double biceps",
    "bodybuilding most muscular pose", "bodybuilder posing competition",
    "physique competition stage", "barbell squat gym", "deadlift competition",
    "bench press gym", "dumbbell lateral raise",
]

# map query hints to Forma pose labels
POSE_HINTS = {
    "front double biceps": "Front Double Biceps",
    "lat spread": "Front Lat Spread",
    "side chest": "Side Chest",
    "back double biceps": "Back Double Biceps",
    "most muscular": "Most Muscular (Crab)",
}


def _api(params: dict) -> dict:
    qs = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{qs}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def search_images(query: str, limit: int) -> list[dict]:
    """Search file namespace; return [{title, url, author, license, page_url}]."""
    out, cont = [], {}
    while len(out) < limit:
        params = {
            "action": "query", "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
            "gsrlimit": min(50, limit), "prop": "imageinfo",
            "iiprop": "url|extmetadata|size", **cont,
        }
        data = _api(params)
        pages = (data.get("query") or {}).get("pages", {})
        for p in pages.values():
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata") or {}
            lic = (meta.get("LicenseShortName") or {}).get("value", "")
            if not ALLOWED_LICENSES.match(lic.strip()):
                continue
            if (ii.get("width") or 0) < 300 or (ii.get("height") or 0) < 400:
                continue
            author = re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "unknown")).strip()
            out.append({
                "title": p.get("title", ""), "url": ii.get("url"),
                "author": author[:120], "license": lic,
                "page_url": ii.get("descriptionurl", ""),
            })
        cont = data.get("continue") or {}
        if not cont:
            break
        time.sleep(0.5)  # be polite
    return out[:limit]


def harvest(queries: list[str], out_dir: Path, limit: int) -> int:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else []
    seen = {r["image"] for r in labels}
    attr_path = out_dir / "attribution.csv"
    new_attr = not attr_path.exists()
    n = 0
    with attr_path.open("a", newline="", encoding="utf-8") as af:
        aw = csv.writer(af)
        if new_attr:
            aw.writerow(["image", "author", "license", "source_url"])
        for q in queries:
            pose = next((v for k, v in POSE_HINTS.items() if k in q.lower()), None)
            for item in search_images(q, limit):
                name = re.sub(r"[^A-Za-z0-9._-]", "_", item["title"].replace("File:", ""))[:120]
                if name in seen or not item["url"]:
                    continue
                try:
                    req = urllib.request.Request(item["url"], headers=UA)
                    data = urllib.request.urlopen(req, timeout=60).read()
                    (img_dir / name).write_bytes(data)
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {name}: {e}")
                    continue
                labels.append({"image": name, "pose": pose,
                               "bf_method": None})  # unlabeled: label in studio
                aw.writerow([name, item["author"], item["license"], item["page_url"]])
                seen.add(name)
                n += 1
                time.sleep(0.3)
            print(f"[{q}] total so far: {n}")
    labels_path.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_commons")
    ap.add_argument("--query", action="append", default=None,
                    help="repeatable; defaults to a curated pose/exercise set")
    ap.add_argument("--limit", type=int, default=25, help="max images per query")
    args = ap.parse_args(argv)
    queries = args.query or DEFAULT_QUERIES
    n = harvest(queries, Path(args.out), args.limit)
    print(f"harvested {n} freely licensed images -> {args.out} "
          f"(attribution.csv written; label them in the studio UI)")


if __name__ == "__main__":
    main()
