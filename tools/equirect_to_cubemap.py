"""Convert an equirectangular HDR image to a vertical-strip cubemap PNG.

Output: W×6W image with faces stacked top-to-bottom in order:
  +X, -X, +Y, -Y, +Z, -Z  (standard cubemap face order for wgpu/Bevy).

Usage:
  uv run python tools/equirect_to_cubemap.py INPUT.hdr OUTPUT.png [FACE_SIZE]
"""

import sys
import numpy as np
from pathlib import Path

import cv2


def equirect_to_cubemap(equirect: np.ndarray, face_size: int) -> np.ndarray:
    """Convert equirectangular image to 6-face vertical strip."""
    h, w, c = equirect.shape

    # Generate UV grid for one face [0, face_size) -> [-1, 1]
    u = np.linspace(-1, 1, face_size, dtype=np.float32)
    v = np.linspace(-1, 1, face_size, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    faces = []
    # Face definitions: each gives (x, y, z) from (uu, vv)
    # +X, -X, +Y, -Y, +Z, -Z
    face_transforms = [
        lambda u, v: ( np.ones_like(u), -v, -u),  # +X (right)
        lambda u, v: (-np.ones_like(u), -v,  u),  # -X (left)
        lambda u, v: ( u,  np.ones_like(u),  v),  # +Y (top)
        lambda u, v: ( u, -np.ones_like(u), -v),  # -Y (bottom)
        lambda u, v: ( u, -v,  np.ones_like(u)),   # +Z (front)
        lambda u, v: (-u, -v, -np.ones_like(u)),   # -Z (back)
    ]

    for transform in face_transforms:
        x, y, z = transform(uu, vv)
        # Spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(y / r, -1, 1))  # polar angle from +Y
        phi = np.arctan2(x, z)                     # azimuth

        # Map to equirectangular UV
        eq_u = (phi / (2 * np.pi) + 0.5) * (w - 1)
        eq_v = (theta / np.pi) * (h - 1)

        # Bilinear sample
        eq_u = np.clip(eq_u, 0, w - 1)
        eq_v = np.clip(eq_v, 0, h - 1)

        u0 = np.floor(eq_u).astype(np.int32)
        v0 = np.floor(eq_v).astype(np.int32)
        u1 = np.minimum(u0 + 1, w - 1)
        v1 = np.minimum(v0 + 1, h - 1)

        fu = (eq_u - u0).astype(np.float32)
        fv = (eq_v - v0).astype(np.float32)

        fu = fu[:, :, np.newaxis]
        fv = fv[:, :, np.newaxis]

        face = (
            equirect[v0, u0] * (1 - fu) * (1 - fv) +
            equirect[v0, u1] * fu * (1 - fv) +
            equirect[v1, u0] * (1 - fu) * fv +
            equirect[v1, u1] * fu * fv
        )
        faces.append(face)

    return np.concatenate(faces, axis=0)  # Stack vertically: 6*face_size x face_size


def tonemap_aces(hdr: np.ndarray) -> np.ndarray:
    """Simple ACES filmic tonemap for HDR -> LDR conversion."""
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    x = hdr
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(mapped, 0, 1)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} INPUT.hdr OUTPUT.png [FACE_SIZE]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    face_size = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    print(f"Loading {input_path}...")
    equirect = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    equirect = cv2.cvtColor(equirect, cv2.COLOR_BGR2RGB).astype(np.float32)
    print(f"  Equirectangular shape: {equirect.shape}, range: [{equirect.min():.3f}, {equirect.max():.3f}]")

    print(f"Converting to cubemap (face_size={face_size})...")
    cubemap = equirect_to_cubemap(equirect, face_size)
    print(f"  Cubemap strip shape: {cubemap.shape}")

    ext = Path(output_path).suffix.lower()
    if ext == ".png":
        # Tonemap HDR to LDR for PNG
        exposure = 2.0  # Boost night sky brightness
        ldr = tonemap_aces(cubemap * exposure)
        out = cv2.cvtColor((ldr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, out)
    elif ext in (".hdr", ".exr"):
        out = cv2.cvtColor(cubemap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, out)
    else:
        print(f"Unknown output format: {ext}")
        sys.exit(1)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
