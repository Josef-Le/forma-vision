"""Scrape bodybuilding contest RESULTS (placings) — facts, not photos.

    python -m formavision.datasets.contest_results --olympia --out olympia.csv

Pulls Mr. Olympia results tables from Wikipedia (CC BY-SA; placings are facts).
Output: year, placing, competitor. Use as ordinal labels ONLY when you have
rights to matching imagery — this module deliberately does not download photos.
"""
from __future__ import annotations

import argparse
import csv


def scrape_olympia(out_path: str) -> int:
    import pandas as pd
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    url = "https://en.wikipedia.org/wiki/List_of_male_professional_bodybuilders"
    rows = []
    # Year pages carry a Results table; iterate a safe recent span.
    for year in range(1990, 2026):
        page = f"https://en.wikipedia.org/wiki/{year}_Mr._Olympia"
        try:
            tables = pd.read_html(page)
        except Exception:
            continue
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("place" in c for c in cols) and any("name" in c or "competitor" in c for c in cols):
                place_col = t.columns[[i for i, c in enumerate(cols) if "place" in c][0]]
                name_col = t.columns[[i for i, c in enumerate(cols) if "name" in c or "competitor" in c][0]]
                for _, r in t.iterrows():
                    place, name = r[place_col], r[name_col]
                    if str(place).strip() and str(name).strip():
                        rows.append({"year": year, "place": str(place)[:6], "competitor": str(name)[:80]})
                break
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "place", "competitor"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--olympia", action="store_true")
    ap.add_argument("--out", default="olympia_results.csv")
    args = ap.parse_args(argv)
    if args.olympia:
        n = scrape_olympia(args.out)
        print(f"wrote {n} placings -> {args.out}")
    else:
        print("nothing selected; use --olympia")


if __name__ == "__main__":
    main()
