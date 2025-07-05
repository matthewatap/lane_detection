#!/bin/bash
# Easy run script for lane detection

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${RED}Error: GOOGLE_API_KEY not set${NC}"
    echo "Please run: export GOOGLE_API_KEY='your_key_here'"
    exit 1
fi

# Show usage
show_help() {
    echo "Lane Detection Runner"
    echo ""
    echo "Usage:"
    echo "  ./run.sh                    # Process all images in examples/"
    echo "  ./run.sh --count 1          # Process just 1 image"
    echo "  ./run.sh --count 5          # Process first 5 images"
    echo "  ./run.sh --image test.jpg   # Process specific image"
    echo "  ./run.sh --help             # Show this help"
    echo ""
    echo "Options:"
    echo "  --count N         Process only N images"
    echo "  --image PATH      Process specific image"
    echo "  --input-dir DIR   Use different input directory (default: examples)"
    echo "  --output-dir DIR  Use specific output directory"
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found. Run setup.sh first.${NC}"
fi

# Check if SAM model exists
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo -e "${YELLOW}SAM model not found. Downloading...${NC}"
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

# Run the main script with all arguments passed through
echo -e "${GREEN}Starting lane detection...${NC}"
python src/enhanced_sam_detection.py "$@"
