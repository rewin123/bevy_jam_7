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
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models.base import SingleFrameStyleModel, SequenceStyleModel
from models.model5 import Model5
from models.model5_seq import Model5Seq
from models.reconet import ReCoNet
from datasets.coco import COCODataset
from datasets.video_flow import SintelDataset, FlyingChairsDataset
from datasets.balanced_loader import BalancedMultiLoader
from utils.vgg16 import VGG16Features
from utils.losses import (
    preprocess_for_vgg,
    gram_matrix,
    content_loss,
    style_loss,
    total_variation_loss,
    pixel_loss,
    output_temporal_loss,
    feature_temporal_loss,
)
from utils.optical_flow import occlusion_mask_from_flow, warp_optical_flow
from utils.transforms import FlowAwareResize, FlowAwareRandomHorizontalFlip, FlowAwareToTensor

# =============================================================================
# CONFIG — edit these variables to configure training
# =============================================================================


class Timeline:
    """Piecewise-linear parameter schedule.

    Takes a list of (step, value) keyframes sorted by step.
    Linearly interpolates between keyframes.
    Before the first keyframe — holds first value.
    After the last keyframe — holds last value.

    Static value: Timeline([(0, val)])
    Warm-up:      Timeline([(0, 0), (1000, 0), (2000, 5.0)])
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

# VGG16 weights
VGG16_WEIGHTS = "data/vgg16.pth"

# Output
OUTPUT_DIR = "outputs"
TENSORBOARD_DIR = "runs"

# Training
TOTAL_STEPS = 50_000
BATCH_SIZE = 4
NUM_WORKERS = 4

# All timelines: list of (step, value) with linear interpolation
LR             = Timeline([(0, 1e-3)])
GRAD_MAX_NORM  = Timeline([(0, 10.0)])

bw = 100

style_scale = 1e3 # Style has different scaling, we need this parameter 
content_scale = 1e-1

# Loss weights (all losses are .mean()-normalized, raw values ~O(1))
CONTENT_WEIGHT = Timeline([(bw, 0.0), (bw + 100, 1.0 * content_scale)])
STYLE_WEIGHT   = Timeline([(bw, 0.0), (bw + 100, 10.0 * style_scale)])
TV_WEIGHT      = Timeline([(400, 0.0), (500, 1.0)])
PIXEL_WEIGHT   = Timeline([(0, 1.0), (bw, 1.0), (bw + 100, 0.1)])
LAMBDA_F       = Timeline([(0, 0.0), (bw, 0.0), (bw + 100, 1.0)])
LAMBDA_O       = Timeline([(0, 0.0), (bw, 0.0), (bw + 100, 2.0)])

# Dataset sampling weights (relative, normalized within each group)
VIDEO_DATASET_WEIGHTS = {"sintel": 1.0, "chairs": 1.0}
STATIC_DATASET_WEIGHTS = {"coco": 1.0, "game": 1.0}

# Checkpoints & Logging
CHECKPOINT_INTERVAL = 200
LOG_INTERVAL = 25
IMAGE_INTERVAL = 100

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
    """Resolve STYLE_IMAGES config to a list of image paths.

    Accepts: single path (str), list of paths, or directory path.
    """
    if isinstance(style_input, list):
        return style_input
    if os.path.isdir(style_input):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(
            os.path.join(style_input, f)
            for f in os.listdir(style_input)
            if os.path.splitext(f)[1].lower() in exts
        )
    return [style_input]


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
    h, w = seq_desc.resolution
    print(f"Model: {MODEL_TYPE} ({sum(p.numel() for p in model.parameters())} params)")
    print(f"Resolution: {h}x{w}")

    # VGG16 feature extractor (frozen)
    vgg = VGG16Features(VGG16_WEIGHTS).to(device)

    # Precompute style Gram matrices (averaged over all style images)
    style_paths = resolve_style_paths(STYLE_IMAGES)
    assert style_paths, f"No style images found in {STYLE_IMAGES}"
    style_grams = None
    with torch.no_grad():
        for sp in style_paths:
            img = load_style_image(sp, seq_desc.resolution).to(device)
            feats = vgg(preprocess_for_vgg(img))
            grams = [gram_matrix(f) for f in feats]
            if style_grams is None:
                style_grams = grams
            else:
                style_grams = [a + b for a, b in zip(style_grams, grams)]
        style_grams = [g / len(style_paths) for g in style_grams]
    style_img = load_style_image(style_paths[0], seq_desc.resolution).to(device)
    print(f"Style images ({len(style_paths)}): {[os.path.basename(p) for p in style_paths]}")

    # Build datasets
    flow_transform = transforms.Compose([
        FlowAwareResize(w, h),
        FlowAwareRandomHorizontalFlip(),
        FlowAwareToTensor(),
    ])

    sintel = SintelDataset(SINTEL_DIR, transform=flow_transform)
    chairs = FlyingChairsDataset(FLYING_CHAIRS_DIR, transform=flow_transform)
    coco = COCODataset(COCO_DIR, (h, w))

    try:
        game = COCODataset(GAME_DIR, (h, w))
    except FileNotFoundError:
        print(f"Warning: Game dataset not found at {GAME_DIR}, skipping")
        game = None

    video_datasets = []
    video_weights = []
    for name, ds, weight_key in [("sintel", sintel, "sintel"), ("chairs", chairs, "chairs")]:
        if len(ds) > 0:
            video_datasets.append((name, ds))
            video_weights.append(VIDEO_DATASET_WEIGHTS[weight_key])
    assert video_datasets, "No video datasets found"

    static_datasets = []
    static_weights = []
    if len(coco) > 0:
        static_datasets.append(("coco", coco))
        static_weights.append(STATIC_DATASET_WEIGHTS["coco"])
    if game is not None and len(game) > 0:
        static_datasets.append(("game", game))
        static_weights.append(STATIC_DATASET_WEIGHTS["game"])
    assert static_datasets, "No static datasets found"

    video_loader = BalancedMultiLoader(
        video_datasets, BATCH_SIZE, NUM_WORKERS, video_weights,
    )
    static_loader = BalancedMultiLoader(
        static_datasets, BATCH_SIZE, NUM_WORKERS, static_weights,
    )

    for name, ds in video_datasets + static_datasets:
        print(f"  {name}: {len(ds)} samples")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR.at(0))

    # Derive style name for output directory
    from datetime import datetime
    if len(style_paths) == 1:
        style_name = os.path.splitext(os.path.basename(style_paths[0]))[0]
    else:
        if isinstance(STYLE_IMAGES, str) and os.path.isdir(STYLE_IMAGES):
            style_name = os.path.basename(os.path.normpath(STYLE_IMAGES))
        else:
            style_name = f"{len(style_paths)}styles"

    style_output_dir = os.path.join(OUTPUT_DIR, style_name)
    os.makedirs(style_output_dir, exist_ok=True)

    # TensorBoard
    run_name = f"{MODEL_TYPE}_{style_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(TENSORBOARD_DIR, run_name)
    writer = SummaryWriter(run_dir)
    print(f"Style output dir: {style_output_dir}")
    print(f"TensorBoard run: {run_dir}")

    best_loss = float("inf")

    pbar = tqdm(range(TOTAL_STEPS), desc="Training")
    for global_step in pbar:

        # Sample all timeline values for this step
        w_content = CONTENT_WEIGHT.at(global_step)
        w_style = STYLE_WEIGHT.at(global_step)
        w_tv = TV_WEIGHT.at(global_step)
        w_pixel = PIXEL_WEIGHT.at(global_step)
        w_lambda_f = LAMBDA_F.at(global_step)
        w_lambda_o = LAMBDA_O.at(global_step)
        lr = LR.at(global_step)
        grad_max_norm = GRAD_MAX_NORM.at(global_step)

        # Update learning rate
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # =====================================================================
        # Video batch: content + style + TV + pixel + temporal
        # =====================================================================
        vid_ds_name, vid_batch = video_loader.next()

        frame = vid_batch["frame"].to(device)
        prev_frame = vid_batch["previous_frame"].to(device)
        flow = vid_batch["optical_flow"].to(device)
        rev_flow = vid_batch["reverse_optical_flow"].to(device)
        motion_bound = vid_batch["motion_boundaries"].to(device)

        occ_mask = occlusion_mask_from_flow(flow, rev_flow, motion_bound)

        feat_t = model.encode(frame)
        out_t = model.decode(feat_t)
        feat_t1 = model.encode(prev_frame)
        out_t1 = model.decode(feat_t1)

        vgg_in_t = vgg(preprocess_for_vgg(frame))
        vgg_out_t = vgg(preprocess_for_vgg(out_t))
        vgg_in_t1 = vgg(preprocess_for_vgg(prev_frame))
        vgg_out_t1 = vgg(preprocess_for_vgg(out_t1))

        vid_c_loss = w_content * (
            content_loss(vgg_out_t[2], vgg_in_t[2])
            + content_loss(vgg_out_t1[2], vgg_in_t1[2])
        )
        vid_s_loss = w_style * (
            style_loss(vgg_out_t, style_grams)
            + style_loss(vgg_out_t1, style_grams)
        )
        vid_tv_loss = w_tv * (
            total_variation_loss(out_t) + total_variation_loss(out_t1)
        )
        vid_p_loss = w_pixel * (
            pixel_loss(out_t, frame) + pixel_loss(out_t1, prev_frame)
        )

        internal_in_t = frame * 2 - 1
        internal_in_t1 = prev_frame * 2 - 1
        internal_out_t = out_t * 2 - 1
        internal_out_t1 = out_t1 * 2 - 1

        f_temp = w_lambda_f * feature_temporal_loss(
            feat_t, feat_t1, rev_flow, occ_mask
        )
        o_temp = w_lambda_o * output_temporal_loss(
            internal_in_t, internal_in_t1,
            internal_out_t, internal_out_t1,
            rev_flow, occ_mask,
        )

        video_total = vid_c_loss + vid_s_loss + vid_tv_loss + vid_p_loss + f_temp + o_temp

        # =====================================================================
        # Static batch: content + style + TV + pixel
        # =====================================================================
        sta_ds_name, sta_batch = static_loader.next()

        images = sta_batch.to(device)
        sta_feat = model.encode(images)
        sta_output = model.decode(sta_feat)

        vgg_in_s = vgg(preprocess_for_vgg(images))
        vgg_out_s = vgg(preprocess_for_vgg(sta_output))

        sta_c_loss = w_content * content_loss(vgg_out_s[2], vgg_in_s[2])
        sta_s_loss = w_style * style_loss(vgg_out_s, style_grams)
        sta_tv_loss = w_tv * total_variation_loss(sta_output)
        sta_p_loss = w_pixel * pixel_loss(sta_output, images)

        static_total = sta_c_loss + sta_s_loss + sta_tv_loss + sta_p_loss

        # =====================================================================
        # Backward + optimize
        # =====================================================================
        total = video_total + static_total

        optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_max_norm)
        optimizer.step()

        pbar.set_postfix(loss=f"{total.item():.2f}", grad=f"{grad_norm:.2f}")

        # TensorBoard logging
        if global_step % LOG_INTERVAL == 0:
            losses_dict = {
                "video/content": vid_c_loss.item(),
                "video/style": vid_s_loss.item(),
                "video/tv": vid_tv_loss.item(),
                "video/pixel": vid_p_loss.item(),
                "video/feat_temporal": f_temp.item(),
                "video/out_temporal": o_temp.item(),
                "video/total": video_total.item(),
                "static/content": sta_c_loss.item(),
                "static/style": sta_s_loss.item(),
                "static/tv": sta_tv_loss.item(),
                "static/pixel": sta_p_loss.item(),
                "static/total": static_total.item(),
                "total": total.item(),
                "weight/content": w_content,
                "weight/style": w_style,
                "weight/tv": w_tv,
                "weight/pixel": w_pixel,
                "weight/lambda_f": w_lambda_f,
                "weight/lambda_o": w_lambda_o,
                "weight/lr": lr,
                "weight/grad_max_norm": grad_max_norm,
                "grad_norm": grad_norm.item(),
            }
            for name, val in losses_dict.items():
                writer.add_scalar(f"loss/{name}", val, global_step)

        # Sample images
        if global_step % IMAGE_INTERVAL == 0:
            with torch.no_grad():
                model.eval()

                # Static: content vs styled
                grid = torchvision.utils.make_grid(
                    [images[0].cpu(), sta_output[0].cpu()], nrow=2
                )
                writer.add_image("train/content_vs_styled", grid, global_step)

                # Video: frame pair + styled outputs
                _b = 0
                grid_frames = torchvision.utils.make_grid(
                    [frame[_b].cpu(), prev_frame[_b].cpu(),
                     out_t[_b].cpu(), out_t1[_b].cpu()],
                    nrow=2,
                )
                writer.add_image("seq/frames_and_outputs", grid_frames, global_step)

                # Optical flow visualization
                def flow_to_rgb(f):
                    """Convert [H,W,2] flow to [3,H,W] RGB."""
                    fx, fy = f[..., 0], f[..., 1]
                    mag = (fx**2 + fy**2).sqrt()
                    max_mag = mag.max().clamp(min=1.0)
                    angle = torch.atan2(fy, fx)
                    h_val = (angle / (2 * 3.14159) + 0.5) % 1.0
                    s_val = (mag / max_mag).clamp(0, 1)
                    v_val = torch.ones_like(s_val)
                    hi = (h_val * 6).long() % 6
                    f_frac = h_val * 6 - hi.float()
                    p = v_val * (1 - s_val)
                    q = v_val * (1 - f_frac * s_val)
                    t_val = v_val * (1 - (1 - f_frac) * s_val)
                    r = torch.where(hi == 0, v_val, torch.where(hi == 1, q, torch.where(hi == 2, p, torch.where(hi == 3, p, torch.where(hi == 4, t_val, v_val)))))
                    g = torch.where(hi == 0, t_val, torch.where(hi == 1, v_val, torch.where(hi == 2, v_val, torch.where(hi == 3, q, torch.where(hi == 4, p, p)))))
                    b = torch.where(hi == 0, p, torch.where(hi == 1, p, torch.where(hi == 2, t_val, torch.where(hi == 3, v_val, torch.where(hi == 4, v_val, q)))))
                    return torch.stack([r, g, b], dim=0)

                flow_fwd_rgb = flow_to_rgb(flow[_b].cpu())
                flow_rev_rgb = flow_to_rgb(rev_flow[_b].cpu())
                grid_flow = torchvision.utils.make_grid(
                    [flow_fwd_rgb, flow_rev_rgb], nrow=2,
                )
                writer.add_image("seq/flow_fwd_rev", grid_flow, global_step)

                # Occlusion mask
                occ_vis = occ_mask[_b].cpu().expand(3, -1, -1)
                writer.add_image("seq/occlusion_mask", occ_vis, global_step)

                # Warped frames
                warped_prev_input = warp_optical_flow(prev_frame[:1], rev_flow[:1])
                warped_prev_output = warp_optical_flow(out_t1[:1], rev_flow[:1])
                grid_warp = torchvision.utils.make_grid(
                    [frame[_b].cpu(), warped_prev_input[_b].cpu(),
                     out_t[_b].cpu(), warped_prev_output[_b].cpu()],
                    nrow=2,
                )
                writer.add_image("seq/curr_vs_warped_prev", grid_warp, global_step)

                # Temporal loss internals
                int_in_t = frame[_b:_b+1] * 2 - 1
                int_in_t1 = prev_frame[_b:_b+1] * 2 - 1
                int_out_t = out_t[_b:_b+1] * 2 - 1
                int_out_t1 = out_t1[_b:_b+1] * 2 - 1
                rev_f = rev_flow[_b:_b+1]
                occ_m = occ_mask[_b:_b+1]

                input_diff = int_in_t - warp_optical_flow(int_in_t1, rev_f)
                output_diff = int_out_t - warp_optical_flow(int_out_t1, rev_f)
                luminance = (input_diff[:, 0] * 0.2126 + input_diff[:, 1] * 0.7152 + input_diff[:, 2] * 0.0722).unsqueeze(1)
                temporal_error = occ_m * (output_diff - luminance)

                def to_vis(t):
                    """Normalize tensor to [0,1] for visualization."""
                    t = t[0].cpu()
                    if t.shape[0] == 1:
                        t = t.expand(3, -1, -1)
                    t_min, t_max = t.min(), t.max()
                    if t_max - t_min > 1e-8:
                        t = (t - t_min) / (t_max - t_min)
                    else:
                        t = t * 0
                    return t

                grid_temporal = torchvision.utils.make_grid(
                    [to_vis(input_diff), to_vis(luminance),
                     to_vis(output_diff), to_vis(temporal_error)],
                    nrow=2,
                )
                writer.add_image("seq/temporal_internals", grid_temporal, global_step)

                # Temporal error heatmap
                err_sq = temporal_error.pow(2).sum(dim=1, keepdim=True)
                err_sq = err_sq[0, 0].cpu()
                err_max = err_sq.max().clamp(min=1e-8)
                err_norm = (err_sq / err_max)
                r_ch = (err_norm * 2).clamp(0, 1)
                g_ch = (err_norm * 2 - 1).clamp(0, 1)
                b_ch = torch.zeros_like(err_norm)
                heatmap = torch.stack([r_ch, g_ch, b_ch], dim=0)
                writer.add_image("seq/temporal_error_heatmap", heatmap, global_step)

                model.train()

        # Save checkpoint every N steps
        if global_step > 0 and global_step % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(style_output_dir, f"checkpoint_{global_step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best model
        if total.item() < best_loss:
            best_loss = total.item()
            torch.save(model.state_dict(), os.path.join(style_output_dir, "best_model.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(style_output_dir, "final_model.pth"))
    writer.close()
    print(f"Training complete. Models saved to {style_output_dir}/")


if __name__ == "__main__":
    train()
