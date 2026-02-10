"""Style transfer training pipeline.

Configuration: edit variables in the CONFIG section below.
Run: cd training && uv run python run_pipeline.py
"""

import os
import sys

# Resolve all paths relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models.base import SingleFrameStyleModel, SequenceStyleModel
from models.model5 import Model5
from models.model5_seq import Model5Seq
from models.reconet import ReCoNet
from datasets.coco import COCODataset
from datasets.video_flow import (
    SintelDataset,
    FlyingChairsDataset,
    InfiniteSamplerWrapper,
)
from utils.vgg16 import VGG16Features
from utils.losses import (
    preprocess_for_vgg,
    gram_matrix,
    content_loss,
    style_loss,
    total_variation_loss,
    output_temporal_loss,
    feature_temporal_loss,
)
from utils.optical_flow import occlusion_mask_from_flow
from utils.transforms import FlowAwareResize, FlowAwareRandomHorizontalFlip, FlowAwareToTensor

# =============================================================================
# CONFIG — edit these variables to configure training
# =============================================================================

# Model: "model5" (single-frame, ~20K params) or "reconet" (sequence-frame, ~1.68M)
MODEL_TYPE = "model5_seq"

# Style image path (single style per training run)
STYLE_IMAGE = os.path.join(_PROJECT_DIR, "assets/styles/kandinskiy.jpg")

# Dataset paths
COCO_DIR = "data/coco2017/val2017"
SINTEL_DIR = "data/sintel"
FLYING_CHAIRS_DIR = "data/FlyingChairs_release"

# VGG16 weights (downloaded by download_data.sh)
VGG16_WEIGHTS = "data/vgg16.pth"

# Output
OUTPUT_DIR = "outputs"
TENSORBOARD_DIR = "runs"

# Training
EPOCHS = 20000
BATCH_SIZE = 4*2
LR = 1e-3
NUM_WORKERS = 4

# Single-frame loss weights (Model5 paper: beta/alpha = 2.5e4)
ALPHA = 1.0       # content loss weight
# BETA = 2.5e4      # style loss weight
BETA = 10.0
GAMMA = 1e-6      # total variation weight

# Sequence-frame loss weights (ReCoNet paper)
CONTENT_WEIGHT = 2e4
STYLE_WEIGHT = 1e5
TV_WEIGHT = 1e-5
LAMBDA_F = 1e5    # feature temporal loss weight
LAMBDA_O = 2e5    # output temporal loss weight

# Loss warm-up schedule (sequence-frame only)
# Each weight ramps linearly from 0 to full over its range of steps.
# Content is always at full weight; the others phase in sequentially.
WARMUP_STYLE   = (200, 400)    # steps 0..100: STYLE_WEIGHT ramps 0→1
WARMUP_LAMBDA_F = (400, 600) # steps 100..200: LAMBDA_F ramps 0→1
WARMUP_LAMBDA_O = (600, 800) # steps 200..300: LAMBDA_O ramps 0→1

# Logging
LOG_INTERVAL = 25      # log losses every N steps
IMAGE_INTERVAL = 100   # save sample images every N steps

# =============================================================================


def warmup_factor(step: int, start: int, end: int) -> float:
    """Linear ramp from 0.0 to 1.0 over [start, end) steps."""
    if step < start:
        return 0.0
    if step >= end:
        return 1.0
    return (step - start) / (end - start)


def build_model(model_type: str):
    if model_type == "model5":
        return Model5()
    elif model_type == "model5_seq":
        return Model5Seq()
    elif model_type == "reconet":
        return ReCoNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_style_image(path: str, resolution: tuple[int, int]) -> torch.Tensor:
    """Load style image as [1,3,H,W] tensor in [0,1]."""
    h, w = resolution
    transform = transforms.Compose([
        transforms.Resize(max(h, w)),
        transforms.CenterCrop((h, w)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = build_model(MODEL_TYPE)
    model = model.to(device)
    model.train()

    seq_desc = model.image_sequence_needs()
    print(f"Model: {MODEL_TYPE} ({sum(p.numel() for p in model.parameters())} params)")
    print(f"Sequence desc: {seq_desc}")

    # VGG16 feature extractor (frozen)
    vgg = VGG16Features(VGG16_WEIGHTS).to(device)

    # Precompute style Gram matrices
    style_img = load_style_image(STYLE_IMAGE, seq_desc.resolution).to(device)
    with torch.no_grad():
        style_vgg_features = vgg(preprocess_for_vgg(style_img))
        style_grams = [gram_matrix(f) for f in style_vgg_features]
    print(f"Style image: {STYLE_IMAGE}")

    # Build dataset + dataloader
    if isinstance(model, SingleFrameStyleModel):
        dataset = COCODataset(COCO_DIR, seq_desc.resolution)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Dataset: COCO ({len(dataset)} images)")
    elif isinstance(model, SequenceStyleModel):
        from torchvision import transforms as T

        h, w = seq_desc.resolution
        flow_transform = T.Compose([
            FlowAwareResize(w, h),
            FlowAwareRandomHorizontalFlip(),
            FlowAwareToTensor(),
        ])

        sintel = SintelDataset(SINTEL_DIR, transform=flow_transform)
        chairs = FlyingChairsDataset(FLYING_CHAIRS_DIR, transform=flow_transform)
        dataset = torch.utils.data.ConcatDataset([sintel, chairs])
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=InfiniteSamplerWrapper(dataset),
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Dataset: Sintel ({len(sintel)}) + FlyingChairs ({len(chairs)}) = {len(dataset)}")
    else:
        raise TypeError(f"Unknown model base type: {type(model)}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # TensorBoard + output dirs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writer = SummaryWriter(TENSORBOARD_DIR)

    # Test image for visualization
    test_img = style_img  # use style image as test (always available)

    best_loss = float("inf")
    global_step = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:

            # --- Single-frame training ---
            if isinstance(model, SingleFrameStyleModel):
                images = batch.to(device)
                output = model(images)

                # VGG features
                vgg_input = vgg(preprocess_for_vgg(images))
                vgg_output = vgg(preprocess_for_vgg(output))

                # Content loss at relu3_3 (index 2)
                c_loss = ALPHA * content_loss(vgg_output[2], vgg_input[2])
                # Style loss at all 4 layers
                s_loss = BETA * style_loss(vgg_output, style_grams)
                # Total variation
                tv_loss = GAMMA * total_variation_loss(output)

                total = c_loss + s_loss + tv_loss

                losses_dict = {
                    "content": c_loss.item(),
                    "style": s_loss.item(),
                    "tv": tv_loss.item(),
                    "total": total.item(),
                }

            # --- Sequence-frame training ---
            elif isinstance(model, SequenceStyleModel):
                frame = batch["frame"].to(device)
                prev_frame = batch["previous_frame"].to(device)
                flow = batch["optical_flow"].to(device)
                rev_flow = batch["reverse_optical_flow"].to(device)
                motion_bound = batch["motion_boundaries"].to(device)

                # Occlusion mask
                occ_mask = occlusion_mask_from_flow(flow, rev_flow, motion_bound)

                # Encode/decode both frames
                feat_t = model.encode(frame)
                out_t = model.decode(feat_t)
                feat_t1 = model.encode(prev_frame)
                out_t1 = model.decode(feat_t1)

                # VGG features for both frames
                vgg_in_t = vgg(preprocess_for_vgg(frame))
                vgg_out_t = vgg(preprocess_for_vgg(out_t))
                vgg_in_t1 = vgg(preprocess_for_vgg(prev_frame))
                vgg_out_t1 = vgg(preprocess_for_vgg(out_t1))

                # Warm-up multipliers
                w_style = warmup_factor(global_step, *WARMUP_STYLE)
                w_lambda_f = warmup_factor(global_step, *WARMUP_LAMBDA_F)
                w_lambda_o = warmup_factor(global_step, *WARMUP_LAMBDA_O)

                # Content loss (relu3_3 = index 2 for ReCoNet)
                c_loss = CONTENT_WEIGHT * (
                    content_loss(vgg_out_t[2], vgg_in_t[2])
                    + content_loss(vgg_out_t1[2], vgg_in_t1[2])
                )

                # Style loss at all 4 layers (warm-up)
                s_loss = STYLE_WEIGHT * w_style * (
                    style_loss(vgg_out_t, style_grams)
                    + style_loss(vgg_out_t1, style_grams)
                )

                # Total variation
                tv_loss = TV_WEIGHT * (
                    total_variation_loss(out_t) + total_variation_loss(out_t1)
                )

                # Temporal losses (warm-up)
                internal_in_t = frame * 2 - 1
                internal_in_t1 = prev_frame * 2 - 1
                internal_out_t = out_t * 2 - 1
                internal_out_t1 = out_t1 * 2 - 1

                f_temp = LAMBDA_F * w_lambda_f * feature_temporal_loss(
                    feat_t, feat_t1, rev_flow, occ_mask
                )
                o_temp = LAMBDA_O * w_lambda_o * output_temporal_loss(
                    internal_in_t, internal_in_t1,
                    internal_out_t, internal_out_t1,
                    rev_flow, occ_mask,
                )

                total = c_loss + s_loss + tv_loss + f_temp + o_temp

                losses_dict = {
                    "content": c_loss.item(),
                    "style": s_loss.item(),
                    "tv": tv_loss.item(),
                    "feat_temporal": f_temp.item(),
                    "out_temporal": o_temp.item(),
                    "total": total.item(),
                    "warmup/style": w_style,
                    "warmup/lambda_f": w_lambda_f,
                    "warmup/lambda_o": w_lambda_o,
                }

            # Backward + optimize
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_loss += total.item()
            epoch_steps += 1
            pbar.set_postfix(loss=f"{total.item():.2f}")

            # TensorBoard logging
            if global_step % LOG_INTERVAL == 0:
                for name, val in losses_dict.items():
                    writer.add_scalar(f"loss/{name}", val, global_step)

            # Sample images
            if global_step % IMAGE_INTERVAL == 0:
                with torch.no_grad():
                    model.eval()
                    if isinstance(model, SingleFrameStyleModel):
                        sample_out = model(images[:1])
                        grid = torchvision.utils.make_grid(
                            [images[0].cpu(), sample_out[0].cpu()], nrow=2
                        )
                    else:
                        sample_out = model(frame[:1])
                        grid = torchvision.utils.make_grid(
                            [frame[0].cpu(), sample_out[0].cpu()], nrow=2
                        )
                    writer.add_image("train/content_vs_styled", grid, global_step)
                    model.train()

            # Save best model
            if total.item() < best_loss:
                best_loss = total.item()
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))

            global_step += 1

            # For sequence datasets with infinite sampler, limit per epoch
            if isinstance(model, SequenceStyleModel) and epoch_steps >= len(dataset) // BATCH_SIZE:
                break

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} — avg loss: {avg_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    writer.close()
    print(f"Training complete. Models saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    train()
