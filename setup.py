#!/bin/bash
# RunPod setup script for SAM ViT-H Lane Detection

echo "=== RunPod SAM Lane Detection Setup ==="

# Update system
apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SAM and dependencies
pip install segment-anything
pip install opencv-python
pip install supervision
pip install google-genai
pip install Pillow
pip install numpy

# Download SAM checkpoints
echo "Downloading SAM ViT-H checkpoint..."
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Optional: Install GroundingDINO for better text-to-mask
echo "Installing GroundingDINO for advanced text grounding..."
pip install groundingdino-py

# Create working directory
mkdir -p /workspace/lane_detection
cd /workspace/lane_detection

echo "=== Setup Complete ==="
echo "To run:"
echo "1. Set your GOOGLE_API_KEY environment variable"
echo "2. Upload your images to /workspace/lane_detection"
echo "3. Run: python lane_detection_sam.py"
