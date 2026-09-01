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

# default query set: mandatory poses, divisions, and common exercises.
# Broad on purpose — results are deduped by filename and license-filtered.
DEFAULT_QUERIES = [
    "bodybuilding front double biceps", "bodybuilding lat spread",
    "bodybuilding side chest pose", "bodybuilding back double biceps",
    "bodybuilding most muscular pose", "bodybuilding abdominal thigh pose",
    "bodybuilder posing competition", "bodybuilding competition stage",
    "classic physique competition", "men's physique competition",
    "women's physique competition", "female bodybuilder competition",
    "bikini fitness competition stage", "bodybuilder flexing",
    "muscular man shirtless gym", "physique athlete",
    "barbell squat gym", "deadlift competition", "powerlifting competition",
    "bench press gym", "dumbbell lateral raise", "weightlifting training gym",
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
    """API call with polite backoff — a 429/5xx never kills the harvest."""
    qs = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{qs}", headers=UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                wait = int(e.headers.get("Retry-After") or 0) or (8 * (2 ** attempt))
                print(f"  API {e.code} — backing off {wait}s")
                time.sleep(wait)
                continue
            print(f"  API gave up ({e.code})")
            return {"query": {"pages": {}}}
        except Exception as e:  # noqa: BLE001
            print(f"  API error: {e}")
            return {"query": {"pages": {}}}
    return {"query": {"pages": {}}}


def _item_from_page(p: dict) -> dict | None:
    """License-filter + size-filter one API page object into an item, or None."""
    ii = (p.get("imageinfo") or [{}])[0]
    meta = ii.get("extmetadata") or {}
    lic = (meta.get("LicenseShortName") or {}).get("value", "")
    if not ALLOWED_LICENSES.match(lic.strip()):
        return None
    if (ii.get("width") or 0) < 300 or (ii.get("height") or 0) < 400:
        return None
    author = re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "unknown")).strip()
    # Wikimedia asks bulk users to fetch THUMBNAILS, not originals — and we
    # train at 384px, so a 1024px thumb is lossless for our purposes.
    return {"title": p.get("title", ""), "url": ii.get("thumburl") or ii.get("url"),
            "author": author[:120], "license": lic,
            "page_url": ii.get("descriptionurl", "")}


def search_images(query: str, limit: int) -> list[dict]:
    """Search file namespace; return [{title, url, author, license, page_url}]."""
    out, cont = [], {}
    while len(out) < limit:
        params = {
            "action": "query", "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
            "gsrlimit": min(50, limit), "prop": "imageinfo",
            "iiprop": "url|extmetadata|size", "iiurlwidth": "1024", **cont,
        }
        data = _api(params)
        for p in ((data.get("query") or {}).get("pages", {})).values():
            item = _item_from_page(p)
            if item:
                out.append(item)
        cont = data.get("continue") or {}
        if not cont:
            break
        time.sleep(0.5)  # be polite
    return out[:limit]


# Category traversal is where the real yield is — Commons organizes physique
# photos under categories far more thoroughly than text search finds them.
DEFAULT_CATEGORIES = [
    "Category:Bodybuilding poses", "Category:Male bodybuilders",
    "Category:Female bodybuilders", "Category:Bodybuilding competitions",
    "Category:Bodybuilding", "Category:Powerlifting competitions",
    "Category:Weight training",
]


def category_images(cat: str, budget: int, max_depth: int = 2) -> list[dict]:
    """BFS a category tree (depth-capped), license-filtering every file."""
    out, seen_cats, queue = [], {cat}, [(cat, 0)]
    while queue and len(out) < budget:
        current, depth = queue.pop(0)
        cont = {}
        while len(out) < budget:
            params = {"action": "query", "generator": "categorymembers",
                      "gcmtitle": current, "gcmtype": "file|subcat",
                      "gcmlimit": "100", "prop": "imageinfo",
                      "iiprop": "url|extmetadata|size", "iiurlwidth": "1024", **cont}
            try:
                data = _api(params)
            except Exception as e:  # noqa: BLE001
                print(f"  category {current}: {e}")
                break
            for p in ((data.get("query") or {}).get("pages", {})).values():
                title = p.get("title", "")
                if title.startswith("Category:"):
                    if depth < max_depth and title not in seen_cats:
                        seen_cats.add(title)
                        queue.append((title, depth + 1))
                else:
                    item = _item_from_page(p)
                    if item:
                        out.append(item)
            cont = data.get("continue") or {}
            if not cont:
                break
            time.sleep(0.3)
    return out[:budget]


def harvest(queries: list[str], out_dir: Path, limit: int,
            categories: list[str] | None = None, cat_budget: int = 0) -> int:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else []
    seen = {r["image"] for r in labels}
    attr_path = out_dir / "attribution.csv"
    new_attr = not attr_path.exists()
    n = 0
    throttled = 0
    stop = False
    with attr_path.open("a", newline="", encoding="utf-8") as af:
        aw = csv.writer(af)
        if new_attr:
            aw.writerow(["image", "author", "license", "source_url"])
        sources = [("search", q, limit) for q in queries]
        if cat_budget > 0:
            per_cat = max(20, cat_budget // max(1, len(categories or DEFAULT_CATEGORIES)))
            sources += [("category", c, per_cat) for c in (categories or DEFAULT_CATEGORIES)]
        for kind, q, lim in sources:
            pose = next((v for k, v in POSE_HINTS.items() if k in q.lower()), None)
            items = search_images(q, lim) if kind == "search" else category_images(q, lim)
            for item in items:
                name = re.sub(r"[^A-Za-z0-9._-]", "_", item["title"].replace("File:", ""))[:120]
                if name in seen or not item["url"]:
                    continue
                data = None
                for attempt in range(2):
                    try:
                        req = urllib.request.Request(item["url"], headers=UA)
                        data = urllib.request.urlopen(req, timeout=60).read()
                        throttled = 0
                        break
                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            throttled += 1
                            if attempt == 0:
                                wait = int(e.headers.get("Retry-After") or 0) or 30
                                print(f"  429 on {name} — waiting {wait}s")
                                time.sleep(wait)
                                continue
                        print(f"  skip {name}: {e}")
                        break
                    except Exception as e:  # noqa: BLE001
                        print(f"  skip {name}: {e}")
                        break
                if throttled >= 3:
                    print("  rate limited 3x in a row — stopping this harvest politely; a rerun continues where this left off")
                    stop = True
                    break
                if data is None:
                    continue
                (img_dir / name).write_bytes(data)
                labels.append({"image": name, "pose": pose,
                               "bf_method": None})  # unlabeled: label in studio
                aw.writerow([name, item["author"], item["license"], item["page_url"]])
                seen.add(name)
                n += 1
                time.sleep(1.0)
            print(f"[{q}] total so far: {n}")
            if stop:
                break
    labels_path.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_commons")
    ap.add_argument("--query", action="append", default=None,
                    help="repeatable; defaults to a curated pose/exercise set")
    ap.add_argument("--limit", type=int, default=25, help="max images per query")
    ap.add_argument("--category", action="append", default=None,
                    help="repeatable; Commons categories to walk (defaults used when --cat-budget > 0)")
    ap.add_argument("--cat-budget", type=int, default=400,
                    help="total images to pull from category traversal (0 = search only)")
    args = ap.parse_args(argv)
    queries = args.query or DEFAULT_QUERIES
    n = harvest(queries, Path(args.out), args.limit, args.category, args.cat_budget)
    print(f"harvested {n} freely licensed images -> {args.out} "
          f"(attribution.csv written; label them in the studio UI)")


if __name__ == "__main__":
    main()
