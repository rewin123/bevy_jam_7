"""Stylize video using trained model (PyTorch or ONNX).

Usage:
    cd training && uv run python process_video.py --model outputs/best_model.pth --video input.mp4 --outfile output.mp4
    cd training && uv run python process_video.py --model model.onnx --video input.mp4 --outfile output.mp4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.video_io import VideoReader, VideoWriter


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame (H,W,3 uint8) to target size."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description="Stylize video")
    parser.add_argument("--model", required=True, help=".pth or .onnx model path")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--outfile", required=True, help="Output video file")
    parser.add_argument(
        "--model-type",
        default="model5",
        choices=["model5", "reconet"],
        help="Model architecture (for .pth files)",
    )
    parser.add_argument("--size", type=int, default=512, help="Inference resolution")
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS")
    args = parser.parse_args()

    is_onnx = args.model.endswith(".onnx")

    # Setup model
    if is_onnx:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.model)
        print(f"ONNX model: {args.model}")

        def stylize_frame(frame_np: np.ndarray) -> np.ndarray:
            """frame_np: [H,W,3] uint8 -> [H,W,3] uint8"""
            tensor = frame_np.astype(np.float32) / 255.0
            tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # [1,3,H,W]
            out = sess.run(["output"], {"input": tensor})[0]
            out = np.clip(out[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            return out
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.model_type == "model5":
            from models.model5 import Model5
            model = Model5()
        else:
            from models.reconet import ReCoNet
            model = ReCoNet()

        model.load_state_dict(
            torch.load(args.model, map_location="cpu", weights_only=True)
        )
        model.eval().to(device)
        print(f"PyTorch model: {args.model} on {device}")

        state = None

        def stylize_frame(frame_np: np.ndarray) -> np.ndarray:
            nonlocal state
            tensor = torch.from_numpy(frame_np).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
            out, state = model.inference_frame(tensor, state)
            out = torch.clamp(out[0].cpu() * 255, 0, 255).byte()
            return out.permute(1, 2, 0).numpy()

    # Process video
    reader = VideoReader(args.video, fps=args.fps)
    inf_size = args.size
    print(f"Input: {reader.width}x{reader.height} @ {reader.fps} fps")
    print(f"Inference size: {inf_size}x{inf_size}")

    writer = VideoWriter(
        args.outfile,
        input_width=reader.width,
        input_height=reader.height,
        input_fps=reader.fps,
    )

    frame_count = 0
    with writer:
        for frame in tqdm(reader, desc="Processing"):
            # Resize to inference resolution
            resized = resize_frame(frame, inf_size, inf_size)

            # Stylize
            styled = stylize_frame(resized)

            # Resize back to original resolution
            output = resize_frame(styled, reader.width, reader.height)
            writer.write(output)
            frame_count += 1

    print(f"Output: {args.outfile} ({frame_count} frames)")


if __name__ == "__main__":
    main()
