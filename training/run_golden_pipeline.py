"""Supervised style transfer training with AdaIN as teacher model.

Instead of unsupervised Gram-matrix losses, this pipeline uses AdaIN
(Adaptive Instance Normalization) to generate ground-truth stylized images,
then trains Model5Seq to match them via direct supervision.

Temporal stability is achieved by training on video frame pairs with temporal
losses, but golden supervision is applied ONLY to frame_t (not frame_t-1)
to avoid propagating the teacher's per-frame inconsistencies.

Two modes:
    --precompute    Generate golden cache and exit
    (default)       Check cache, then train

Run: cd training && uv run python run_golden_pipeline.py
     cd training && uv run python run_golden_pipeline.py --precompute
"""

import argparse
import os
import sys

# Resolve all paths relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models.base import SequenceStyleModel
from models.model5 import Model5
from models.model5_seq import Model5Seq
from models.reconet import ReCoNet
from datasets.golden_datasets import (
    GoldenCOCODataset,
    GoldenSintelDataset,
    GoldenFlyingChairsDataset,
    GoldenFlowAwareResize,
    GoldenFlowAwareRandomHorizontalFlip,
    GoldenFlowAwareToTensor,
)
from datasets.balanced_loader import BalancedMultiLoader
from utils.vgg16 import VGG16Features
from utils.losses import (
    preprocess_for_vgg,
    content_loss,
    total_variation_loss,
    pixel_loss,
    output_temporal_loss,
    feature_temporal_loss,
)
from utils.optical_flow import occlusion_mask_from_flow, warp_optical_flow


# =============================================================================
# CONFIG
# =============================================================================

class Timeline:
    """Piecewise-linear parameter schedule.

    Takes a list of (step, value) keyframes sorted by step.
    Linearly interpolates between keyframes.
    """

    def __init__(self, keyframes: list[tuple[int, float]]):
        assert keyframes, "Timeline must have at least one keyframe"
        self.steps = [s for s, _ in keyframes]
        self.values = [v for _, v in keyframes]

    def at(self, step: int) -> float:
        if step <= self.steps[0]:
            return self.values[0]
        if step >= self.steps[-1]:
            return self.values[-1]
        for i in range(len(self.steps) - 1):
            if self.steps[i] <= step < self.steps[i + 1]:
                t = (step - self.steps[i]) / (self.steps[i + 1] - self.steps[i])
                return self.values[i] + t * (self.values[i + 1] - self.values[i])
        return self.values[-1]

    def __repr__(self) -> str:
        kf = ", ".join(f"{s}:{v:.4g}" for s, v in zip(self.steps, self.values))
        return f"Timeline([{kf}])"


# Model
MODEL_TYPE = "model5_seq"

# Style images: single path, list of paths, or directory
STYLE_IMAGES = os.path.join(_PROJECT_DIR, "assets/styles/candy.jpg")

# Dataset paths
COCO_DIR = "data/coco2017/val2017"
GAME_DIR = "data/game"
SINTEL_DIR = "data/sintel"
FLYING_CHAIRS_DIR = "data/FlyingChairs_release"

# VGG16 weights (for perceptual loss during training, NOT NNST's VGG)
VGG16_WEIGHTS = "data/vgg16.pth"

# Golden cache
GOLDEN_CACHE_DIR = "data/golden_cache"

# AdaIN teacher parameters
ADAIN_ALPHA = 0.7  # 0.0 = no style, 1.0 = full stylization

# Pre-computation limits (0 = all images)
COCO_LIMIT = 1000
CHAIRS_LIMIT = 1000

# Output
OUTPUT_DIR = "outputs"
TENSORBOARD_DIR = "runs"

# Training
TOTAL_STEPS = 50_000
BATCH_SIZE = 4
NUM_WORKERS = 4

# Loss weight schedules
bw = 200  # warm-up boundary

LR                       = Timeline([(0, 1e-4), (30_000, 1e-5)])
GRAD_MAX_NORM            = Timeline([(0, 10.0)])

GOLDEN_PIXEL_WEIGHT      = Timeline([(0, 5.0)])
GOLDEN_PERCEPTUAL_WEIGHT = Timeline([(0, 1.0)])
TV_WEIGHT                = Timeline([(0, 0.1)])
LAMBDA_F                 = Timeline([(0, 0.0), (bw, 0.0), (bw + 200, 1.0)])
LAMBDA_O                 = Timeline([(0, 0.0), (bw, 0.0), (bw + 200, 2.0)])

# Dataset sampling weights
VIDEO_DATASET_WEIGHTS = {"sintel": 1.0, "chairs": 1.0}
STATIC_DATASET_WEIGHTS = {"coco": 1.0, "game": 1.0}

# Checkpoints & Logging
CHECKPOINT_INTERVAL = 200
LOG_INTERVAL = 25
IMAGE_INTERVAL = 100


# =============================================================================
# HELPERS
# =============================================================================

def build_model(model_type: str):
    if model_type == "model5":
        return Model5()
    elif model_type == "model5_seq":
        return Model5Seq()
    elif model_type == "reconet":
        return ReCoNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def resolve_style_paths(style_input) -> list[str]:
    """Resolve STYLE_IMAGES config to a list of image paths."""
    if isinstance(style_input, list):
        return sorted(style_input)
    if os.path.isdir(style_input):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(
            os.path.join(style_input, f)
            for f in os.listdir(style_input)
            if os.path.splitext(f)[1].lower() in exts
        )
    return [style_input]


def collect_content_paths(
    coco_dir: str,
    sintel_dir: str,
    chairs_dir: str,
    game_dir: str,
    coco_limit: int = 0,
    chairs_limit: int = 0,
) -> list[str]:
    """Collect all unique content image paths across datasets."""
    paths = []

    # COCO / Game — flat image folders
    for d in [coco_dir, game_dir]:
        if os.path.isdir(d):
            imgs = sorted(
                os.path.join(d, f) for f in os.listdir(d)
                if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
            )
            paths.extend(imgs)

    # Sintel — all frames across all scenes
    frames_dir = os.path.join(sintel_dir, "training", "clean")
    if os.path.isdir(frames_dir):
        for scene in sorted(os.listdir(frames_dir)):
            scene_dir = os.path.join(frames_dir, scene)
            if os.path.isdir(scene_dir):
                for f in sorted(os.listdir(scene_dir)):
                    if f.endswith(".png"):
                        paths.append(os.path.join(scene_dir, f))

    # FlyingChairs — img1 and img2
    chairs_data = os.path.join(chairs_dir, "data")
    if os.path.isdir(chairs_data):
        chair_imgs = sorted(
            os.path.join(chairs_data, f) for f in os.listdir(chairs_data)
            if f.endswith("_img1.ppm") or f.endswith("_img2.ppm")
        )
        paths.extend(chair_imgs)

    # Deduplicate (in case of overlaps)
    paths = sorted(set(paths))

    # Apply limits
    if coco_limit > 0 or chairs_limit > 0:
        filtered = []
        coco_count = 0
        chairs_count = 0
        for p in paths:
            if coco_dir and os.path.abspath(p).startswith(os.path.abspath(coco_dir)):
                if coco_limit > 0 and coco_count >= coco_limit:
                    continue
                coco_count += 1
            if chairs_dir and os.path.abspath(p).startswith(os.path.abspath(chairs_dir)):
                if chairs_limit > 0 and chairs_count >= chairs_limit:
                    continue
                chairs_count += 1
            filtered.append(p)
        paths = filtered

    return paths


def _golden_path_for_precompute(
    content_path: str, data_root: str, cache_root: str,
) -> str:
    """Map content path to golden cache path."""
    rel = os.path.relpath(content_path, data_root)
    base = os.path.splitext(rel)[0] + ".png"
    return os.path.join(cache_root, base)


# =============================================================================
# PRE-COMPUTATION
# =============================================================================

def precompute_golden(style_paths: list[str], resolution: tuple[int, int]):
    """Generate AdaIN golden stylizations for all training images."""
    from utils.adain_wrapper import AdaINStylizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stylizer = AdaINStylizer(
        style_paths=style_paths,
        device=device,
        alpha=ADAIN_ALPHA,
    )

    style_hash = stylizer.style_hash()
    cache_root = os.path.join(GOLDEN_CACHE_DIR, style_hash)
    print(f"Style hash: {style_hash}")
    print(f"Cache root: {cache_root}")

    # Collect all content paths
    all_paths = collect_content_paths(
        COCO_DIR, SINTEL_DIR, FLYING_CHAIRS_DIR, GAME_DIR,
        coco_limit=COCO_LIMIT, chairs_limit=CHAIRS_LIMIT,
    )
    print(f"Total content images: {len(all_paths)}")

    # The data root is the common ancestor of all dataset dirs
    data_root = "data"

    # Filter to only uncomputed images
    todo = []
    for p in all_paths:
        out_path = _golden_path_for_precompute(p, data_root, cache_root)
        if not os.path.exists(out_path):
            todo.append((p, out_path))

    print(f"Already cached: {len(all_paths) - len(todo)}")
    print(f"To compute: {len(todo)}")

    if not todo:
        print("All golden images already cached.")
        stylizer.cleanup()
        return cache_root

    h, w = resolution
    resize_transform = transforms.Compose([
        transforms.Resize((h, w)),  # exact resize to match training transforms
        transforms.ToTensor(),
    ])

    for i, (content_path, out_path) in enumerate(tqdm(todo, desc="Pre-computing golden")):
        try:
            img = Image.open(content_path).convert("RGB")
            content_tensor = resize_transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

            # Round-robin style assignment
            style_idx = i % len(style_paths)

            golden = stylizer.stylize(content_tensor, style_index=style_idx)

            # Save as PNG
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            golden_np = (golden[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
            Image.fromarray(golden_np).save(out_path)

        except Exception as e:
            print(f"\nError processing {content_path}: {e}")
            continue

        # Periodic GPU cache cleanup
        if i % 10 == 0:
            torch.cuda.empty_cache()

    stylizer.cleanup()
    print(f"Pre-computation complete. Cache: {cache_root}")
    return cache_root


# =============================================================================
# TRAINING
# =============================================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = build_model(MODEL_TYPE)
    model = model.to(device)
    model.train()

    seq_desc = model.image_sequence_needs()
    h, w = seq_desc.resolution
    print(f"Model: {MODEL_TYPE} ({sum(p.numel() for p in model.parameters())} params)")
    print(f"Resolution: {h}x{w}")

    # Determine golden cache directory
    style_paths = resolve_style_paths(STYLE_IMAGES)
    assert style_paths, f"No style images found in {STYLE_IMAGES}"

    # Compute style hash without loading full AdaIN model
    import hashlib
    sh = hashlib.sha256()
    sh.update(b"adain_v1")
    for sp in sorted(style_paths):
        with open(sp, "rb") as f:
            sh.update(f.read())
    sh.update(f"alpha={ADAIN_ALPHA}".encode())
    style_hash = sh.hexdigest()[:12]

    cache_root = os.path.join(GOLDEN_CACHE_DIR, style_hash)
    data_root = "data"
    print(f"Golden cache: {cache_root}")

    if not os.path.isdir(cache_root):
        print("ERROR: Golden cache not found. Run with --precompute first.")
        sys.exit(1)

    # VGG16 feature extractor (frozen) — for perceptual losses during training
    vgg = VGG16Features(VGG16_WEIGHTS).to(device)

    # Build golden datasets
    # Map dataset dirs to their golden subdirs
    def golden_dir_for(dataset_dir: str) -> str:
        rel = os.path.relpath(dataset_dir, data_root)
        return os.path.join(cache_root, rel)

    # Video transforms (with golden support)
    flow_transform = transforms.Compose([
        GoldenFlowAwareResize(w, h),
        GoldenFlowAwareRandomHorizontalFlip(),
        GoldenFlowAwareToTensor(),
    ])

    # Build video datasets
    sintel_golden = golden_dir_for(SINTEL_DIR)
    chairs_golden = golden_dir_for(FLYING_CHAIRS_DIR)

    sintel = GoldenSintelDataset(
        SINTEL_DIR, sintel_golden, transform=flow_transform,
    )
    chairs = GoldenFlyingChairsDataset(
        FLYING_CHAIRS_DIR, chairs_golden, transform=flow_transform,
    )

    # Build static datasets
    coco_golden = golden_dir_for(COCO_DIR)
    coco = GoldenCOCODataset(COCO_DIR, coco_golden, (h, w))

    game = None
    try:
        game_golden = golden_dir_for(GAME_DIR)
        game = GoldenCOCODataset(GAME_DIR, game_golden, (h, w))
    except FileNotFoundError:
        print(f"Warning: Game golden dataset not available, skipping")

    # Assemble loaders
    video_datasets = []
    video_weights = []
    for name, ds, weight_key in [("sintel", sintel, "sintel"), ("chairs", chairs, "chairs")]:
        if len(ds) > 0:
            video_datasets.append((name, ds))
            video_weights.append(VIDEO_DATASET_WEIGHTS[weight_key])

    static_datasets = []
    static_weights = []
    if len(coco) > 0:
        static_datasets.append(("coco", coco))
        static_weights.append(STATIC_DATASET_WEIGHTS["coco"])
    if game is not None and len(game) > 0:
        static_datasets.append(("game", game))
        static_weights.append(STATIC_DATASET_WEIGHTS["game"])

    assert static_datasets, "No static datasets found with golden pairs"

    if video_datasets:
        video_loader = BalancedMultiLoader(
            video_datasets, BATCH_SIZE, NUM_WORKERS, video_weights,
        )
        has_video = True
    else:
        print("Warning: No video datasets with golden pairs, training without temporal losses")
        has_video = False

    static_loader = BalancedMultiLoader(
        static_datasets, BATCH_SIZE, NUM_WORKERS, static_weights,
    )

    for name, ds in (video_datasets if has_video else []) + static_datasets:
        print(f"  {name}: {len(ds)} samples")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR.at(0))

    # Output directory
    from datetime import datetime
    if len(style_paths) == 1:
        style_name = os.path.splitext(os.path.basename(style_paths[0]))[0]
    else:
        if isinstance(STYLE_IMAGES, str) and os.path.isdir(STYLE_IMAGES):
            style_name = os.path.basename(os.path.normpath(STYLE_IMAGES))
        else:
            style_name = f"{len(style_paths)}styles"

    style_output_dir = os.path.join(OUTPUT_DIR, f"{style_name}_golden")
    os.makedirs(style_output_dir, exist_ok=True)

    # TensorBoard
    run_name = f"{MODEL_TYPE}_{style_name}_golden_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(TENSORBOARD_DIR, run_name)
    writer = SummaryWriter(run_dir)
    print(f"Output dir: {style_output_dir}")
    print(f"TensorBoard: {run_dir}")

    best_loss = float("inf")

    pbar = tqdm(range(TOTAL_STEPS), desc="Training")
    for global_step in pbar:

        # Sample all timeline values
        w_golden_pixel = GOLDEN_PIXEL_WEIGHT.at(global_step)
        w_golden_percep = GOLDEN_PERCEPTUAL_WEIGHT.at(global_step)
        w_tv = TV_WEIGHT.at(global_step)
        w_lambda_f = LAMBDA_F.at(global_step)
        w_lambda_o = LAMBDA_O.at(global_step)
        lr = LR.at(global_step)
        grad_max_norm = GRAD_MAX_NORM.at(global_step)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # =================================================================
        # Video batch: golden on frame_t + temporal on both
        # =================================================================
        if has_video:
            vid_ds_name, vid_batch = video_loader.next()

            frame = vid_batch["frame"].to(device)
            prev_frame = vid_batch["previous_frame"].to(device)
            golden_frame = vid_batch["golden_frame"].to(device)
            flow = vid_batch["optical_flow"].to(device)
            rev_flow = vid_batch["reverse_optical_flow"].to(device)
            motion_bound = vid_batch["motion_boundaries"].to(device)

            occ_mask = occlusion_mask_from_flow(flow, rev_flow, motion_bound)

            # Forward both frames
            feat_t = model.encode(frame)
            out_t = model.decode(feat_t)
            feat_t1 = model.encode(prev_frame)
            out_t1 = model.decode(feat_t1)

            # Golden supervision — ONLY on frame_t
            vid_golden_pixel = w_golden_pixel * pixel_loss(out_t, golden_frame)

            vgg_out_t = vgg(preprocess_for_vgg(out_t))
            vgg_golden_t = vgg(preprocess_for_vgg(golden_frame))
            vid_golden_percep = w_golden_percep * content_loss(
                vgg_out_t[2], vgg_golden_t[2]  # relu3_3
            )

            # Temporal losses — on BOTH frames
            internal_in_t = frame * 2 - 1
            internal_in_t1 = prev_frame * 2 - 1
            internal_out_t = out_t * 2 - 1
            internal_out_t1 = out_t1 * 2 - 1

            f_temp = w_lambda_f * feature_temporal_loss(
                feat_t, feat_t1, rev_flow, occ_mask,
            )
            o_temp = w_lambda_o * output_temporal_loss(
                internal_in_t, internal_in_t1,
                internal_out_t, internal_out_t1,
                rev_flow, occ_mask,
            )

            vid_tv = w_tv * (
                total_variation_loss(out_t) + total_variation_loss(out_t1)
            )

            video_total = vid_golden_pixel + vid_golden_percep + f_temp + o_temp + vid_tv
        else:
            video_total = torch.tensor(0.0, device=device)

        # =================================================================
        # Static batch: golden supervision only
        # =================================================================
        sta_ds_name, sta_batch = static_loader.next()

        content = sta_batch["content"].to(device)
        golden = sta_batch["golden"].to(device)

        sta_feat = model.encode(content)
        sta_output = model.decode(sta_feat)

        sta_golden_pixel = w_golden_pixel * pixel_loss(sta_output, golden)

        vgg_out_s = vgg(preprocess_for_vgg(sta_output))
        vgg_golden_s = vgg(preprocess_for_vgg(golden))
        sta_golden_percep = w_golden_percep * content_loss(
            vgg_out_s[2], vgg_golden_s[2]
        )

        sta_tv = w_tv * total_variation_loss(sta_output)

        static_total = sta_golden_pixel + sta_golden_percep + sta_tv

        # =================================================================
        # Backward + optimize
        # =================================================================
        total = video_total + static_total

        optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_max_norm,
        )
        optimizer.step()

        pbar.set_postfix(loss=f"{total.item():.2f}", grad=f"{grad_norm:.2f}")

        # TensorBoard logging
        if global_step % LOG_INTERVAL == 0:
            losses_dict = {
                "static/golden_pixel": sta_golden_pixel.item(),
                "static/golden_perceptual": sta_golden_percep.item(),
                "static/tv": sta_tv.item(),
                "static/total": static_total.item(),
                "total": total.item(),
                "weight/golden_pixel": w_golden_pixel,
                "weight/golden_perceptual": w_golden_percep,
                "weight/tv": w_tv,
                "weight/lambda_f": w_lambda_f,
                "weight/lambda_o": w_lambda_o,
                "weight/lr": lr,
                "grad_norm": grad_norm.item(),
            }
            if has_video:
                losses_dict.update({
                    "video/golden_pixel": vid_golden_pixel.item(),
                    "video/golden_perceptual": vid_golden_percep.item(),
                    "video/feat_temporal": f_temp.item(),
                    "video/out_temporal": o_temp.item(),
                    "video/tv": vid_tv.item(),
                    "video/total": video_total.item(),
                })
            for name, val in losses_dict.items():
                writer.add_scalar(f"loss/{name}", val, global_step)

        # Sample images
        if global_step % IMAGE_INTERVAL == 0:
            with torch.no_grad():
                model.eval()

                # Static: content vs golden vs model output
                grid = torchvision.utils.make_grid(
                    [content[0].cpu(), golden[0].cpu(), sta_output[0].cpu()],
                    nrow=3,
                )
                writer.add_image("train/content_golden_output", grid, global_step)

                # Error map: |model_output - golden|
                err = (sta_output[0] - golden[0]).abs().cpu()
                err_vis = (err / err.max().clamp(min=1e-8)).clamp(0, 1)
                grid_err = torchvision.utils.make_grid(
                    [golden[0].cpu(), sta_output[0].cpu(), err_vis], nrow=3,
                )
                writer.add_image("train/golden_output_error", grid_err, global_step)

                if has_video:
                    _b = 0

                    # Video: frame pair + golden + outputs
                    grid_vid = torchvision.utils.make_grid(
                        [frame[_b].cpu(), prev_frame[_b].cpu(),
                         golden_frame[_b].cpu(), out_t[_b].cpu(),
                         out_t1[_b].cpu()],
                        nrow=3,
                    )
                    writer.add_image("seq/frames_golden_outputs", grid_vid, global_step)

                    # Occlusion mask
                    occ_vis = occ_mask[_b].cpu().expand(3, -1, -1)
                    writer.add_image("seq/occlusion_mask", occ_vis, global_step)

                    # Warped outputs for temporal visualization
                    warped_prev_output = warp_optical_flow(out_t1[:1], rev_flow[:1])
                    grid_warp = torchvision.utils.make_grid(
                        [out_t[_b].cpu(), warped_prev_output[_b].cpu()],
                        nrow=2,
                    )
                    writer.add_image("seq/curr_vs_warped_prev", grid_warp, global_step)

                    # Temporal error heatmap
                    temp_diff = (out_t[:1] - warped_prev_output).pow(2).sum(dim=1, keepdim=True)
                    temp_diff = temp_diff[0, 0].cpu()
                    t_max = temp_diff.max().clamp(min=1e-8)
                    t_norm = (temp_diff / t_max)
                    r_ch = (t_norm * 2).clamp(0, 1)
                    g_ch = (t_norm * 2 - 1).clamp(0, 1)
                    b_ch = torch.zeros_like(t_norm)
                    heatmap = torch.stack([r_ch, g_ch, b_ch], dim=0)
                    writer.add_image("seq/temporal_error_heatmap", heatmap, global_step)

                model.train()

        # Save checkpoint
        if global_step > 0 and global_step % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(style_output_dir, f"checkpoint_{global_step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best model
        if total.item() < best_loss:
            best_loss = total.item()
            torch.save(
                model.state_dict(),
                os.path.join(style_output_dir, "best_model.pth"),
            )

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(style_output_dir, "final_model.pth"),
    )
    writer.close()
    print(f"Training complete. Models saved to {style_output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Golden supervised style transfer training")
    parser.add_argument(
        "--precompute", action="store_true",
        help="Only generate golden cache, then exit",
    )
    args = parser.parse_args()

    style_paths = resolve_style_paths(STYLE_IMAGES)
    assert style_paths, f"No style images found in {STYLE_IMAGES}"
    print(f"Style images ({len(style_paths)}): {[os.path.basename(p) for p in style_paths]}")

    # Get training resolution from model
    model = build_model(MODEL_TYPE)
    resolution = model.train_resolution()
    del model
    print(f"Training resolution: {resolution[0]}x{resolution[1]}")

    if args.precompute:
        precompute_golden(style_paths, resolution)
    else:
        # Check if golden cache exists, offer to precompute if not
        import hashlib
        sh = hashlib.sha256()
        sh.update(b"adain_v1")
        for sp in sorted(style_paths):
            with open(sp, "rb") as f:
                sh.update(f.read())
        sh.update(f"alpha={ADAIN_ALPHA}".encode())
        style_hash = sh.hexdigest()[:12]
        cache_root = os.path.join(GOLDEN_CACHE_DIR, style_hash)

        if not os.path.isdir(cache_root):
            print(f"\nGolden cache not found at {cache_root}")
            print("Run with --precompute first to generate golden images.")
            print(f"  uv run python run_golden_pipeline.py --precompute")
            sys.exit(1)

        train()


if __name__ == "__main__":
    main()
