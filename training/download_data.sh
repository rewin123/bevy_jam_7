#!/usr/bin/env bash
set -euo pipefail

# Download training data for style transfer models.
# Creates training/data/ directory with:
#   - coco2017/val2017/  (~800MB, 5000 images)
#   - sintel/            (MPI Sintel, ~5.4GB)
#   - FlyingChairs_release/ (~30GB)
#   - vgg16.pth          (VGG16 ImageNet weights, ~553MB)
#
# Usage: cd training && bash download_data.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "================================================"
echo "  Download training data for style transfer"
echo "  Target directory: $DATA_DIR"
echo "================================================"

# --- COCO2017 Validation Set ---
echo ""
echo "=== [1/4] COCO2017 Validation Set ==="
if [ ! -d "coco2017/val2017" ]; then
    mkdir -p coco2017
    echo "Downloading COCO2017 val2017.zip (~800MB)..."
    curl -L -o coco2017/val2017.zip \
        "http://images.cocodataset.org/zips/val2017.zip"
    echo "Extracting..."
    unzip -q coco2017/val2017.zip -d coco2017/
    rm coco2017/val2017.zip
    echo "COCO2017 val: $(ls coco2017/val2017 | wc -l) images"
else
    echo "Already exists, skipping"
fi

# --- MPI Sintel ---
echo ""
echo "=== [2/4] MPI Sintel (training set) ==="
if [ ! -d "sintel/training" ]; then
    mkdir -p sintel
    echo "Downloading MPI-Sintel-complete.zip (~5.4GB)..."
    curl -L -o sintel/sintel.zip \
        "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"
    echo "Extracting..."
    unzip -q sintel/sintel.zip -d sintel/
    rm sintel/sintel.zip
    echo "MPI Sintel downloaded"
else
    echo "Already exists, skipping"
fi

# --- FlyingChairs ---
echo ""
echo "=== [3/4] FlyingChairs ==="
if [ ! -d "FlyingChairs_release" ]; then
    echo "Downloading FlyingChairs.zip (~30GB)..."
    echo "NOTE: This is a large download. You can skip this if you only train single-frame models."
    curl -L -o FlyingChairs.zip \
        "https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip"
    echo "Extracting..."
    unzip -q FlyingChairs.zip
    rm FlyingChairs.zip
    echo "FlyingChairs downloaded"
else
    echo "Already exists, skipping"
fi

# --- VGG16 Weights ---
echo ""
echo "=== [4/4] VGG16 ImageNet Weights ==="
if [ ! -f "vgg16.pth" ]; then
    echo "Downloading VGG16 weights via torchvision..."
    cd "$SCRIPT_DIR"
    uv run python -c "
import torch
from torchvision.models import vgg16, VGG16_Weights
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
torch.save(model.state_dict(), 'data/vgg16.pth')
print('VGG16 weights saved to data/vgg16.pth')
"
    cd "$DATA_DIR"
else
    echo "Already exists, skipping"
fi

echo ""
echo "================================================"
echo "  Download complete!"
echo "================================================"
echo ""
du -sh "$DATA_DIR"/* 2>/dev/null || echo "(empty)"
