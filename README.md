# forma-vision

Fine-tune a vision backbone to score physiques the way Forma's AI judge does —
body-fat %, 15 per-muscle size/definition scores, and an overall judge score —
from your own bulk-uploaded photos and JSON/YAML labels.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A: straight from the Forma app
#   (app → Profile → Export all) — photos pair automatically with your
#   DEXA/caliper logs, judge scores, weight and tape measurements
python -m formavision.prep_forma_export --export forma-export.json --out dataset

# Option B: bulk photos + your own labels file
#   dataset/images/*.jpg  +  dataset/labels.yaml   (schema: formavision/labels.py
#   and examples/labels.example.yaml)

# Validate the whole pipeline in ~1 minute (CPU is fine):
python -m formavision.train --data dataset --dry-run

# Real training (GPU recommended; 30 epochs, ConvNeXt-Tiny @ 384px):
python -m formavision.train --data dataset --epochs 30

# Predict:
python -m formavision.infer --ckpt runs/<run>/best.pt --input newphoto.jpg

# Ship it:
python -m formavision.export_onnx --ckpt runs/<run>/best.pt --out model.onnx
python -m formavision.export_onnx --ckpt runs/<run>/best.pt --out model.mlpackage --coreml
```

## Full UI — no folder-dropping

```bash
pip install -r requirements.txt
python -m formavision.studio --data ./dataset     # → http://localhost:7860
```

The **Studio** is a web UI over the whole pipeline: bulk photo upload, import
of the app's one-tap training bundle (Lab tab → Export bundle), a visual label
editor (pose / bf / method / judge score per photo), the Commons harvester, the
contest-results scraper, and one-button training with a live job log. Run it
locally or in a free GitHub Codespace and open it from your phone.

**Zero-machine option:** push this repo to GitHub and run the included
`.github/workflows/train.yml` from the Actions tab — it harvests free-licensed
images, scrapes contest results, trains on the free CPU runners, and uploads
the model + attribution manifest as build artifacts. (CPU training is slow:
keep epochs ≤ 10 there; use Colab/Kaggle free GPUs for real runs.)

## How the pieces fit

| Module | Job |
|---|---|
| `labels.py` | label schema + validation (json / yaml / jsonl) |
| `dataset.py` | images + masked targets; deterministic train/val split; physique-safe augmentation (no mirror flips — left/right asymmetry is a label) |
| `model.py` | timm backbone → 4 heads; range-bounded outputs; masked SmoothL1 with ground-truth-quality weighting (DEXA teaches 1.0×, BIA 0.55×, AI estimate 0.3×) |
| `train.py` | warmup + cosine LR, head-only freeze phase, AMP, resume, CSV logs, best/last checkpoints |
| `infer.py` | single image / folder → Forma scan-schema JSON |
| `export_onnx.py` | ONNX (server / Android) and Core ML (on-device iOS) |
| `pseudo_label.py` | **distillation labeler** — LLM judge labels the whole dataset (bf/muscles/judge) so the CNN trains at scale before ground truth exists; never overwrites real DEXA/caliper labels |
| `mediapipe_features.py` | bulk MediaPipe pose keypoints → geometric ratios CSV (label QA, tabular baseline, late fusion) |
| `studio.py` | full web UI: upload, label, harvest, scrape, train |
| `datasets/commons.py` | Wikimedia Commons harvester — free licenses only, attribution.csv auto-written |
| `datasets/sources.py` | real obtainable datasets + licenses + fetch commands |
| `datasets/contest_results.py` | contest placings scraper (facts only, no photos) |

## The three-layer data strategy (why this is NOT trained on one person)

1. **Live judge (day one):** Claude's vision model — pretrained at internet
   scale, already accurate; user measurements only calibrate it per person.
2. **Distillation (scale):** harvest free-licensed photos → `pseudo_label.py`
   has the LLM judge label all of them → the CNN learns to reproduce those
   judgments at ~1000x cheaper inference. Pseudo-labels train at 0.3x weight.
3. **Ground truth (the moat):** consented user exports with DEXA/caliper
   readings train at full weight and gradually out-teach the pseudo-labels.

Typical bootstrap: `commons.py --limit 50` (≈500 images) → `pseudo_label.py`
(≈$2-4 of API) → `train.py` → a distilled v0 judge that runs anywhere.

## Where training data actually comes from

Run `python -m formavision.datasets.sources` for the full annotated list. Short
version: public data gives you **tabular body-fat** (Kaggle: the hydrostatic
weighing dataset), **body geometry** (Amazon BodyM — CDLA-permissive silhouettes
+ 14 tape measurements), and **contest placings** (Wikipedia/NPC scorecards —
facts, scrapeable). What does *not* exist publicly is photos paired with
DEXA + judge scores — that dataset is the one your app creates: every user
export is photos with measured bf, judge scores, weight and tape context.
Get explicit training consent in your ToS, weight labels by method quality
(already wired into the loss), and the model improves with every user.

## MediaPipe vs Apple Vision

Both give ~33/19 body keypoints on-device. This repo uses **MediaPipe** (Python
+ web + Android; same landmarks as Forma Lab in the browser). On iOS, swap in
**Apple Vision** (`VNDetectHumanBodyPoseRequest`) for keypoints and run the
exported Core ML model beside it — same features, no extra runtime.

## Training tips that matter for this domain

- 200 labeled photos of *one* person → good personal model; ~3–5k across
  dozens of bodies → generalizes. Until then keep the LLM judge in the loop
  and use this model as the consistency anchor.
- Never mirror-augment; never crop away ankles/shoulders (silhouette is signal).
- Track MAE on **DEXA-labeled samples only** as your headline metric.
- Retrain monthly as exports accumulate; `--resume` from last checkpoint.
