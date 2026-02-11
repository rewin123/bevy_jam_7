"""Preprocess assets for WASM builds.

Converts 16-bit PNG textures embedded in .glb files to 8-bit,
since WebGL lacks TEXTURE_FORMAT_16BIT_NORM support.

Usage: uv run --with Pillow python scripts/preprocess_wasm.py
"""

import io
import json
import struct
import sys
from pathlib import Path

from PIL import Image

ASSETS_DIR = Path(__file__).parent.parent / "assets"
GLB_DIRS = [ASSETS_DIR / "levels"]


def process_glb(path: Path) -> bool:
    """Convert 16-bit textures to 8-bit in a .glb file. Returns True if modified."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"glTF":
            print(f"  SKIP {path.name}: not a valid glTF file")
            return False
        version = struct.unpack("<I", f.read(4))[0]
        _length = struct.unpack("<I", f.read(4))[0]

        json_len = struct.unpack("<I", f.read(4))[0]
        _json_type = f.read(4)
        json_data = json.loads(f.read(json_len))

        bin_len = struct.unpack("<I", f.read(4))[0]
        _bin_type = f.read(4)
        bin_data = bytearray(f.read(bin_len))

    buffer_views = json_data.get("bufferViews", [])
    images = json_data.get("images", [])

    modified = False
    for i, img in enumerate(images):
        bv_idx = img.get("bufferView")
        if bv_idx is None:
            continue
        bv = buffer_views[bv_idx]
        offset = bv.get("byteOffset", 0)
        orig_len = bv["byteLength"]

        png_data = bin_data[offset : offset + orig_len]

        # Check PNG signature
        if png_data[:4] != b"\x89PNG":
            continue

        # Read bit depth from IHDR (offset 24 in PNG = 8 sig + 4 len + 4 type + 8 w/h)
        bit_depth = png_data[24]
        if bit_depth <= 8:
            continue

        name = img.get("name", f"image_{i}")
        print(f"  [{i}] {name}: {bit_depth}-bit -> 8-bit ...", end=" ")

        pil_img = Image.open(io.BytesIO(bytes(png_data)))
        pil_img = pil_img.convert("RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", optimize=True)
        new_png = buf.getvalue()

        if len(new_png) <= orig_len:
            # Replace in-place, zero-pad remainder
            bin_data[offset : offset + orig_len] = new_png + b"\x00" * (
                orig_len - len(new_png)
            )
            bv["byteLength"] = len(new_png)
            modified = True
            print(f"OK ({orig_len} -> {len(new_png)} bytes)")
        else:
            print(f"SKIP (new {len(new_png)} > orig {orig_len}, re-export from Blender)")

    if not modified:
        return False

    # Rewrite GLB
    json_bytes = json.dumps(json_data, separators=(",", ":")).encode("utf-8")
    # Pad to 4-byte alignment
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "
    while len(bin_data) % 4 != 0:
        bin_data += b"\x00"

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)

    with open(path, "wb") as f:
        f.write(b"glTF")
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<I", total_len))
        f.write(struct.pack("<I", len(json_bytes)))
        f.write(b"JSON")
        f.write(json_bytes)
        f.write(struct.pack("<I", len(bin_data)))
        f.write(b"BIN\x00")
        f.write(bytes(bin_data))

    print(f"  -> {path.name} rewritten")
    return True


def main():
    any_modified = False

    for glb_dir in GLB_DIRS:
        if not glb_dir.exists():
            continue
        for glb_path in sorted(glb_dir.glob("*.glb")):
            print(f"Processing {glb_path.relative_to(ASSETS_DIR.parent)} ...")
            if process_glb(glb_path):
                any_modified = True
            else:
                print("  (no changes needed)")

    if any_modified:
        print("\nDone! 16-bit textures converted to 8-bit for WebGL compatibility.")
    else:
        print("\nAll textures already 8-bit compatible.")


if __name__ == "__main__":
    main()
