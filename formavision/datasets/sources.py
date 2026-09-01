"""Registry of real, obtainable data sources for physique-model training —
what each contains, its license posture, and how to fetch it.

    python -m formavision.datasets.sources            # list everything
    python -m formavision.datasets.sources --fetch bodym

THE HONEST PICTURE ON TRAINING DATA
-----------------------------------
There is no public dataset of "physique photos labeled with DEXA body fat and
judge scores" — that pairing is exactly the moat your app builds by collecting
consented user photos + ground-truth logs (the Forma export → prep_forma_export
pipeline). What IS public splits into three useful groups:

1. Tabular body-composition data (no images): trains the measurement→bf model
   that calibrates and sanity-checks the vision model.
2. Body images/measurements datasets: pretraining and geometry supervision.
3. Contest RESULTS (placings): facts, freely scrapeable; the photos from those
   contests are copyrighted — pair placings with photos only under license or
   with your own collected/consented imagery.

Weak-label bootstrap that works today: pull bf-labeled progress photos only
where the poster stated a DEXA/caliper number, treat as bf_method="Caliper"
(0.8 weight) or "estimate" (0.3), and let the masked loss absorb the noise —
respecting each platform's ToS and takedown requests, and never publishing the
photos themselves.
"""
from __future__ import annotations

import argparse

SOURCES = {
    "bodyfat-tabular": dict(
        kind="tabular",
        what="252 men: circumferences + hydrostatic-weighing body fat (the classic Penrose dataset)",
        url="https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset",
        license="public/educational; check Kaggle page",
        fetch="kaggle datasets download -d fedesoriano/body-fat-prediction-dataset",
        use="train measurements->bf regressor; calibrate the app's Navy-formula fallback",
    ),
    "bodyfat-extended": dict(
        kind="tabular",
        what="extended body-fat dataset incl. women",
        url="https://www.kaggle.com/datasets/simonezappatini/body-fat-extended-dataset",
        license="check Kaggle page",
        fetch="kaggle datasets download -d simonezappatini/body-fat-extended-dataset",
        use="same as bodyfat-tabular, adds female coverage",
    ),
    "bodym": dict(
        kind="images+measurements",
        what="Amazon BodyM: ~8k subjects, silhouette images + 14 tape measurements + height/weight",
        url="https://registry.opendata.aws/bodym/",
        license="CDLA-Permissive-1.0 (usable commercially — verify current terms)",
        fetch="aws s3 cp --no-sign-request s3://amazon-bodym ./bodym --recursive",
        use="pretrain the backbone on body geometry; supervise measurement heads",
    ),
    "hf-body-measurements": dict(
        kind="images+measurements",
        what="HuggingFace body-measurements datasets (TrainingDataPro/UniqueData)",
        url="https://huggingface.co/datasets/TrainingDataPro/body-measurements-dataset",
        license="often research/commercial-paid — read each card",
        fetch='python -c "from datasets import load_dataset; load_dataset(\'TrainingDataPro/body-measurements-dataset\')"',
        use="extra geometry supervision",
    ),
    "olympia-results": dict(
        kind="results",
        what="Mr. Olympia placings by year (Wikipedia tables)",
        url="https://en.wikipedia.org/wiki/Mr._Olympia",
        license="CC BY-SA (facts themselves are not copyrightable)",
        fetch="python -m formavision.datasets.contest_results --olympia",
        use="ordinal supervision IF you license or collect matching photos",
    ),
    "npc-scorecards": dict(
        kind="results",
        what="NPC contest scorecards archive (placings per class)",
        url="https://npcnewsonline.com/category/contest-scorecards/",
        license="site content copyrighted; placings are facts — scrape politely, photos are NOT included",
        fetch="manual / custom scraper respecting robots.txt",
        use="same as olympia-results, amateur level (closer to your users)",
    ),
    "own-users": dict(
        kind="images+ground-truth",
        what="YOUR app's consented exports: photos + DEXA/caliper bf + judge scores + tape + weight",
        url="(in-app: Profile -> Export)",
        license="yours, with explicit user consent for model training in your ToS",
        fetch="python -m formavision.prep_forma_export --export forma-export.json --out dataset",
        use="the primary fine-tuning set; everything else is pretraining/calibration",
    ),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", help="print the fetch command for one source")
    args = ap.parse_args(argv)
    if args.fetch:
        s = SOURCES.get(args.fetch)
        if not s:
            print(f"unknown source; options: {', '.join(SOURCES)}")
            return
        print(s["fetch"])
        return
    for name, s in SOURCES.items():
        print(f"\n[{name}] ({s['kind']})\n  {s['what']}\n  url: {s['url']}\n  license: {s['license']}\n  use: {s['use']}\n  fetch: {s['fetch']}")


if __name__ == "__main__":
    main()
