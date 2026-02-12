"""Export trained StyleModel to ONNX with verification.

Usage:
    cd training && uv run python export_onnx.py --model outputs/best_model.pth --out ../assets/models/custom.onnx
    cd training && uv run python export_onnx.py --model outputs/best_model.pth --out model.onnx --model-type reconet
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Export StyleModel to ONNX")
    parser.add_argument("--model", required=True, help="Path to .pth state dict")
    parser.add_argument("--out", required=True, help="Path for output .onnx file")
    parser.add_argument(
        "--model-type",
        default="model5",
        choices=["model5", "model5_seq", "reconet", "stytr_micro"],
        help="Model architecture",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--size", type=int, default=512, help="Export test resolution (square)")
    args = parser.parse_args()

    # 1. Load model
    if args.model_type == "model5":
        from models.model5 import Model5
        model = Model5()
    elif args.model_type == "model5_seq":
        from models.model5_seq import Model5Seq
        model = Model5Seq()
    elif args.model_type == "stytr_micro":
        from models.stytr_micro import StyTRMicro
        model = StyTRMicro()
    else:
        from models.reconet import ReCoNet
        model = ReCoNet()

    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_type} ({param_count} params)")

    # 2. Test forward pass
    dummy = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        pt_out = model(dummy)
    print(f"PyTorch forward: {dummy.shape} -> {pt_out.shape}")
    print(f"  range: [{pt_out.min():.4f}, {pt_out.max():.4f}]")

    # 3. Export to ONNX
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
    )
    print(f"Exported to {out_path}")

    # 4. Verify with onnx checker
    import onnx

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: passed")

    # 5. Verify with onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path))
    ort_out = sess.run(["output"], {"input": dummy.numpy()})[0]

    max_diff = np.abs(pt_out.numpy() - ort_out).max()
    mean_diff = np.abs(pt_out.numpy() - ort_out).mean()
    print(f"PyTorch vs ORT: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if max_diff >= 1e-3:
        print(f"WARNING: max_diff ({max_diff:.6f}) >= 1e-3, model may have issues")
    else:
        print("Verification: PASSED (max_diff < 1e-3)")

    # 6. Test dynamic resolution
    for test_h, test_w in [(256, 256), (384, 512), (128, 192)]:
        dummy2 = np.random.randn(1, 3, test_h, test_w).astype(np.float32)
        ort_out2 = sess.run(["output"], {"input": dummy2})[0]
        print(f"Dynamic resolution: [1,3,{test_h},{test_w}] -> {list(ort_out2.shape)}")

    # 7. Save comparison images
    try:
        comparison_dir = out_path.parent / "verification"
        comparison_dir.mkdir(exist_ok=True)

        # Use a random noise image for visual check
        test_input = torch.clamp(torch.randn(1, 3, args.size, args.size) * 0.2 + 0.5, 0, 1)
        with torch.no_grad():
            pt_result = model(test_input)
        ort_result = sess.run(["output"], {"input": test_input.numpy()})[0]

        def tensor_to_pil(t):
            arr = (t.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        tensor_to_pil(test_input).save(str(comparison_dir / "input.png"))
        tensor_to_pil(pt_result).save(str(comparison_dir / "pytorch_output.png"))
        tensor_to_pil(torch.from_numpy(ort_result)).save(str(comparison_dir / "onnx_output.png"))
        print(f"Comparison images saved to {comparison_dir}/")
    except Exception as e:
        print(f"Could not save comparison images: {e}")

    size_kb = out_path.stat().st_size / 1024
    print(f"\nONNX file: {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
