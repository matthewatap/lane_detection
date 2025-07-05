# lane_detection


A high-quality lane detection system that combines Google's Gemini for semantic understanding with Meta's SAM for pixel-perfect segmentation.

## Features

- 🎯 Semantic lane understanding via Gemini Vision API
- 🖼️ Pixel-perfect segmentation using SAM ViT-H
- 🚗 Detects multiple lane types:
  - White solid lines
  - White dashed lines
  - Yellow solid lines
  - Yellow dashed lines
  - Road edges
- 🚀 GPU acceleration support (CUDA/MPS)
- 📊 Detailed JSON output with statistics

## Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key
- CUDA-capable GPU (recommended) or Apple Silicon Mac

### Setup

### 1. Clone the repository:
bash
git clone https://github.com/YOUR_USERNAME/lane-detection-sam.git
cd lane-detection-sam

### 2. Install dependencies:

bashpip install -r requirements.txt

### 3. Download SAM checkpoint:

bash# The script will auto-download, or manually download:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

### 4. Set your Gemini API key:

bashexport GOOGLE_API_KEY="your_api_key_here"
Usage
Basic Usage
pythonpython src/enhanced_sam_detection.py
Process specific image:
python# Edit the test_image variable in main() or pass as argument
python src/enhanced_sam_detection.py --image "path/to/image.jpg"
Output
The system generates:

*_composite.png - Original image with colored lane overlays
*_overlay.png - Lane masks only
*_results.json - Detailed detection results

### 5. Performance

GPU (RTX 4090): ~2-5 seconds per image
M2 Mac: ~5-10 seconds per image
CPU: ~30-60 seconds per image

Usage
Basic Usage
pythonpython src/enhanced_sam_detection.py
Process specific image:
python# Edit the test_image variable in main() or pass as argument
python src/enhanced_sam_detection.py --image "path/to/image.jpg"
Output
The system generates:

*_composite.png - Original image with colored lane overlays
*_overlay.png - Lane masks only
*_results.json - Detailed detection results

Performance

GPU (RTX 4090): ~2-5 seconds per image
M2 Mac: ~5-10 seconds per image
CPU: ~30-60 seconds per image

RunPod Deployment
See RUNPOD.md for deployment instructions.
License
MIT License - see LICENSE file for details.
EOF

### 6. Create Setup Script

bash
cat > setup.sh << 'EOF'
#!/bin/bash
# Setup script for lane detection system

echo "Setting up Lane Detection with SAM..."

# Check Python version
python3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (with CUDA if available)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch for CPU/MPS..."
    pip install torch torchvision
fi

# Install other requirements
pip install -r requirements.txt

# Download SAM checkpoint if not present
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM ViT-H checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo "Setup complete! Don't forget to set your GOOGLE_API_KEY environment variable."
EOF

chmod +x setup.sh


### Step 7: Create RunPod Documentation
bashmkdir docs
cat > docs/RUNPOD.md << 'EOF'
# RunPod Deployment Guide

## Quick Start

1. Create a RunPod GPU pod (recommended: RTX 4090 or A40)
2. Select PyTorch 2.0+ template
3. SSH into your pod

## Installation

bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/lane-detection-sam.git
cd lane-detection-sam

# Run setup
bash setup.sh

# Set API key
export GOOGLE_API_KEY="your_key_here"

# Upload your images to the pod
# Option 1: Use Jupyter Lab (usually at port 8888)
# Option 2: Use scp from local machine
Running
bashpython src/enhanced_sam_detection.py
Performance Tips

Use persistent storage for model checkpoints
Enable GPU monitoring: watch -n 1 nvidia-smi
For batch processing, consider increasing RAM allocation



### Step 8: Initialize Git Repository

bash
# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Lane detection system with SAM and Gemini"
