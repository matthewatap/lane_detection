#!/bin/bash
# Setup script for lane detection system

echo "Setting up Lane Detection with SAM..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with appropriate backend
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
elif [[ $(uname -s) == "Darwin" ]]; then
    echo "macOS detected, installing PyTorch for MPS..."
    pip install torch torchvision
else
    echo "Installing PyTorch for CPU..."
    pip install torch torchvision
fi

# Install other requirements
pip install -r requirements.txt

# Download SAM checkpoint if not present
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM ViT-H checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your API key: export GOOGLE_API_KEY='your_key_here'"
echo "2. Run example: python run_example.py --image path/to/image.jpg"
