"""Export a Model5Seq with random (untrained) weights to ONNX for testing the burn pipeline."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path

from models.model5_seq import Model5Seq


def main():
    model = Model5Seq()
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model5Seq: {param_count} params")

    out_path = Path(__file__).parent.parent / "assets" / "models" / "styles" / "test_style.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, 288, 512)
    with torch.no_grad():
        pt_out = model(dummy)
    print(f"PyTorch forward: {dummy.shape} -> {pt_out.shape}")
    print(f"  range: [{pt_out.min():.4f}, {pt_out.max():.4f}]")

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
    )
    print(f"Exported to {out_path}")

    import onnx
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: passed")

    # Print ops used
    ops = set()
    for node in onnx_model.graph.node:
        ops.add(node.op_type)
    print(f"ONNX ops used: {sorted(ops)}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path))
    ort_out = sess.run(["output"], {"input": dummy.numpy()})[0]
    max_diff = np.abs(pt_out.numpy() - ort_out).max()
    print(f"PyTorch vs ORT max_diff: {max_diff:.6f}")

    # Also export a fixed-resolution version for burn-import (no dynamic axes)
    fixed_path = Path(__file__).parent.parent / "assets" / "models" / "styles" / "test_style_fixed.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(fixed_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print(f"Fixed-resolution model exported to {fixed_path}")

    size_kb = out_path.stat().st_size / 1024
    print(f"\nONNX file: {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
