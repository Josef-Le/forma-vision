"""Train forma-vision. Point it at a dataset folder and go:

    python -m formavision.train --data ./dataset --epochs 30 --backbone convnext_tiny

Dataset folder = images/ + labels.(yaml|json|jsonl)  (see formavision/labels.py).
Checkpoints + metrics land in --out (default ./runs/<name>).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import PhysiqueDataset
from .model import FormaVision, total_loss


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune a vision backbone on physique data")
    p.add_argument("--data", required=True, help="dataset folder (images/ + labels file)")
    p.add_argument("--out", default=None, help="output dir (default runs/<timestamp>)")
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--freeze-epochs", type=int, default=2,
                   help="epochs to train heads only before unfreezing the backbone")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--dry-run", action="store_true", help="one tiny step to validate the pipeline")
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    abs_bf, n_bf, abs_sz, n_sz, abs_j, n_j = 0.0, 0, 0.0, 0, 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        out = model(x)
        m = batch["bf_mask"].to(device) > 0
        if m.any():
            abs_bf += (out["bf"][m] - batch["bf"].to(device)[m]).abs().sum().item()
            n_bf += int(m.sum())
        mm = batch["muscle_mask"].to(device) > 0
        if mm.any():
            abs_sz += (out["size"][mm] - batch["size"].to(device)[mm]).abs().sum().item()
            n_sz += int(mm.sum())
        jm = batch["judge_mask"].to(device) > 0
        if jm.any():
            abs_j += (out["judge"][jm] - batch["judge"].to(device)[jm]).abs().sum().item()
            n_j += int(jm.sum())
    return {
        "mae_bf": abs_bf / n_bf if n_bf else float("nan"),
        "mae_size": abs_sz / n_sz if n_sz else float("nan"),
        "mae_judge": abs_j / n_j if n_j else float("nan"),
    }


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PhysiqueDataset(args.data, "train", args.img_size, args.val_frac)
    val_ds = PhysiqueDataset(args.data, "val", args.img_size, args.val_frac, augment=False)
    stats = train_ds.label_stats()
    print(f"train: {stats}  val: {val_ds.label_stats()}  device: {device}")
    if stats["with_bf"] + stats["with_muscles"] + stats["with_judge"] == 0:
        raise SystemExit("ABORT: no labeled targets in the train split — every loss would be masked out. "
                         "Label photos in the studio (bf / muscles / judge) before training.")
    if len(val_ds) == 0:
        print("WARNING: empty val split — metrics will be NaN; add more data or lower --val-frac")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=(device == "cuda"), drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    model = FormaVision(args.backbone, pretrained=not args.no_pretrained).to(device)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    back_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([
        {"params": back_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_dl))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_scale(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", best)
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        else:  # older checkpoint: fast-forward the schedule to the right step
            for _ in range(start_epoch * steps_per_epoch):
                sched.step()
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    log_path = out_dir / "log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,loss,mae_bf,mae_size,mae_judge,lr\n")

    for epoch in range(start_epoch, args.epochs):
        frozen = epoch < args.freeze_epochs
        for p in back_params:
            p.requires_grad = not frozen
        model.train()
        running, nb = 0.0, 0
        for batch in train_dl:
            x = batch["image"].to(device)
            for k in ("bf", "bf_mask", "size", "definition", "muscle_mask", "judge", "judge_mask"):
                batch[k] = batch[k].to(device)
            with torch.autocast(device_type=device if device != "mps" else "cpu",
                                enabled=(device == "cuda")):
                out = model(x)
                loss, parts = total_loss(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item(); nb += 1
            if args.dry_run:
                break
        metrics = evaluate(model, val_dl, device) if len(val_ds) else {"mae_bf": float("nan"), "mae_size": float("nan"), "mae_judge": float("nan")}
        avg = running / max(1, nb)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1}/{args.epochs} loss {avg:.3f} | MAE bf {metrics['mae_bf']:.2f} "
              f"size {metrics['mae_size']:.2f} judge {metrics['mae_judge']:.2f} | {'frozen' if frozen else 'full'}")
        with log_path.open("a") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg:.4f}", f"{metrics['mae_bf']:.3f}",
                                    f"{metrics['mae_size']:.3f}", f"{metrics['mae_judge']:.3f}", f"{lr_now:.2e}"])
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
              "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "best": best, "args": vars(args)}
        torch.save(ck, out_dir / "last.pt")
        score = metrics["mae_bf"] if metrics["mae_bf"] == metrics["mae_bf"] else avg
        if score < best:
            best = score
            ck["best"] = best
            torch.save(ck, out_dir / "best.pt")
        if args.dry_run:
            print("dry run OK — pipeline validated end to end")
            break

    (out_dir / "done.json").write_text(json.dumps({"best": best}))
    print(f"done. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
