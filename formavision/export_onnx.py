"""Export a trained checkpoint to ONNX (and optionally Core ML).

    python -m formavision.export_onnx --ckpt runs/xxx/best.pt --out model.onnx
    python -m formavision.export_onnx --ckpt runs/xxx/best.pt --out model.mlpackage --coreml

ONNX runs server-side (onnxruntime) or in-app via onnxruntime-mobile.
Core ML is the path to on-device iOS inference next to Apple Vision's
pose detection (see README: Apple Vision handles keypoints natively;
this model adds the physique scoring).
"""
from __future__ import annotations

import argparse

import torch

from .infer import load_model


class _Wrapper(torch.nn.Module):
    """Flatten the dict output to a stable tuple for export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["bf"], out["size"], out["definition"], out["judge"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--coreml", action="store_true")
    args = ap.parse_args(argv)

    model, train_args = load_model(args.ckpt, "cpu")
    size = args.img_size or train_args.get("img_size", 384)
    wrapper = _Wrapper(model).eval()
    dummy = torch.zeros(1, 3, size, size)

    if args.coreml:
        import coremltools as ct
        traced = torch.jit.trace(wrapper, dummy)
        ml = ct.convert(
            traced,
            inputs=[ct.ImageType(name="image", shape=dummy.shape,
                                 scale=1 / 255.0)],
            outputs=[ct.TensorType(name=n) for n in
                     ("bf", "size", "definition", "judge")],
            minimum_deployment_target=ct.target.iOS16,
        )
        ml.save(args.out)
    else:
        torch.onnx.export(
            wrapper, dummy, args.out,
            input_names=["image"],
            output_names=["bf", "size", "definition", "judge"],
            dynamic_axes={"image": {0: "batch"}},
            opset_version=17,
        )
    print(f"exported -> {args.out}")


if __name__ == "__main__":
    main()
