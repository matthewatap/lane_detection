# 🚦 Lane Detection System

A high-quality lane detection pipeline combining **Google's Gemini Vision API** for semantic understanding and **Meta's SAM ViT-H** for precise segmentation.

---

## 🔧 Features

- 🎯 Semantic lane detection via Gemini Vision API  
- 🖼️ Pixel-perfect segmentation using SAM ViT-H  
- 🛣️ Detects various lane types:
  - White/Yellow — Solid and Dashed  
  - Road edges  
- 🚀 GPU acceleration support (CUDA/MPS)  
- 📊 Detailed JSON output with detection statistics  

---

## 🛠️ Installation

### ✅ Prerequisites

- Python 3.8+  
- Google Gemini API Key  
- CUDA-capable GPU *(recommended)* or Apple Silicon Mac  

### 📥 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lane-detection-sam.git
   cd lane-detection-sam
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SAM checkpoint**
   ```bash
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

4. **Set your Gemini API key**
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

---

## 🚀 Usage

### ▶️ Basic Usage

Run default test image:
```bash
python src/enhanced_sam_detection.py
```

Process a specific image:
```bash
python src/enhanced_sam_detection.py --image "path/to/image.jpg"
```

### 📝 Output Files

Each run generates:

- `*_composite.png` – Original image with colored lane overlays  
- `*_overlay.png` – Lane masks only  
- `*_results.json` – Detailed lane detection results  

---

## ⚡ Performance Benchmarks

| Device         | Avg Time / Image |
|----------------|------------------|
| GPU (RTX 4090) | ~2–5 sec         |
| Apple M2 Mac   | ~5–10 sec        |
| CPU only       | ~30–60 sec       |

---

## 📦 Deployment on RunPod

See [docs/RUNPOD.md](docs/RUNPOD.md) for full deployment instructions.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🔄 Optional Setup Script

To automate installation, create the following script:

```bash
cat > setup.sh << 'EOF'
#!/bin/bash
echo "Setting up Lane Detection with SAM..."

python3 --version
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing CUDA-compatible PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing CPU/MPS-compatible PyTorch..."
    pip install torch torchvision
fi

pip install -r requirements.txt

if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo "Setup complete! Remember to set your GOOGLE_API_KEY."
EOF

chmod +x setup.sh
```

---

## 📁 RunPod Deployment Guide (docs/RUNPOD.md)

```markdown
# 🚀 RunPod Deployment Guide

## Quick Start

1. Launch a GPU pod (Recommended: RTX 4090 or A40)  
2. Use the PyTorch 2.0+ template  
3. SSH into your pod  

## Setup

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/lane-detection-sam.git
cd lane-detection-sam

# Run setup
bash setup.sh

# Set API key
export GOOGLE_API_KEY="your_key_here"
```

## Upload Images

- **Option 1**: Use Jupyter Lab (typically at port 8888)  
- **Option 2**: Use `scp` from your local machine  

## Run the Detector

```bash
python src/enhanced_sam_detection.py
```

## Performance Tips

- Use persistent storage for model checkpoints  
- Enable GPU monitoring:
  ```bash
  watch -n 1 nvidia-smi
  ```
- For batch processing, increase RAM allocation
```

---

## 🧰 Git Initialization

```bash
git init
git add .
git commit -m "Initial commit: Lane detection system with SAM and Gemini"
```
