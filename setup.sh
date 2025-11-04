#!/bin/bash

# Vocal MIDI Generator - Complete Setup Script
# This script sets up the entire development environment

set -e  # Exit on error

echo "========================================="
echo "Vocal MIDI Generator Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

echo "Step 1: Checking prerequisites..."
echo "-----------------------------------"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}✗ CMake not found. Please install CMake 3.22+${NC}"
    exit 1
else
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}✓ CMake found: $CMAKE_VERSION${NC}"
fi

# Check compiler
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    echo -e "${GREEN}✓ GCC found: $GCC_VERSION${NC}"
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1)
    echo -e "${GREEN}✓ Clang found: $CLANG_VERSION${NC}"
else
    echo -e "${RED}✗ No C++ compiler found${NC}"
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ Python3 not found (needed for ML training)${NC}"
fi

# Check Git
if command -v git &> /dev/null; then
    echo -e "${GREEN}✓ Git found${NC}"
else
    echo -e "${RED}✗ Git not found${NC}"
    exit 1
fi

echo ""
echo "Step 2: Setting up JUCE framework..."
echo "-----------------------------------"

# Clone JUCE if not present
if [ ! -d "external/JUCE" ]; then
    echo "Downloading JUCE framework..."
    mkdir -p external
    git clone --depth 1 --branch 7.0.9 \
        https://github.com/juce-framework/JUCE.git external/JUCE
    echo -e "${GREEN}✓ JUCE downloaded${NC}"
else
    echo -e "${GREEN}✓ JUCE already present${NC}"
fi

echo ""
echo "Step 3: Checking ONNX Runtime..."
echo "-----------------------------------"

ONNX_FOUND=false

# Check if ONNX Runtime is installed
if ldconfig -p 2>/dev/null | grep -q onnxruntime; then
    echo -e "${GREEN}✓ ONNX Runtime found in system libraries${NC}"
    ONNX_FOUND=true
elif [ -f "/usr/local/lib/libonnxruntime.so" ]; then
    echo -e "${GREEN}✓ ONNX Runtime found in /usr/local/lib${NC}"
    ONNX_FOUND=true
fi

if [ "$ONNX_FOUND" = false ]; then
    echo -e "${YELLOW}⚠ ONNX Runtime not found${NC}"
    echo "You can install it with:"
    echo "  Ubuntu/Debian: sudo apt-get install libonnxruntime-dev"
    echo "  macOS: brew install onnxruntime"
    echo ""
    echo "Or download from: https://github.com/microsoft/onnxruntime/releases"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 4: Creating build directory..."
echo "-----------------------------------"

if [ -d "build" ]; then
    echo "Build directory exists. Clean rebuild? (y/n)"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf build
        mkdir build
    fi
else
    mkdir build
fi

echo ""
echo "Step 5: Configuring CMake..."
echo "-----------------------------------"

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

echo ""
echo "Step 6: Building plugin..."
echo "-----------------------------------"

# Detect number of CPU cores
if command -v nproc &> /dev/null; then
    CORES=$(nproc)
elif command -v sysctl &> /dev/null; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=4
fi

echo "Building with $CORES parallel jobs..."
cmake --build . --config Release -j$CORES

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

cd ..

echo ""
echo "Step 7: Setting up Python environment (optional)..."
echo "-----------------------------------"

if command -v python3 &> /dev/null; then
    cd ml_training
    
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
        
        echo "Installing Python dependencies..."
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        deactivate
        
        echo -e "${GREEN}✓ Python environment ready${NC}"
    else
        echo -e "${GREEN}✓ Python environment already exists${NC}"
    fi
    
    cd ..
else
    echo -e "${YELLOW}⚠ Skipping Python setup (Python not found)${NC}"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Plugin location:"
echo "  VST3: build/VocalMIDI_artefacts/Release/VST3/"
echo "  Standalone: build/VocalMIDI_artefacts/Release/Standalone/"
echo ""
echo "Next steps:"
echo "1. Test the standalone app:"
echo "   ./build/VocalMIDI_artefacts/Release/Standalone/VocalMIDI"
echo ""
echo "2. Install plugin to system:"
echo "   sudo cmake --install build"
echo ""
echo "3. Train ML models (optional):"
echo "   cd ml_training"
echo "   source venv/bin/activate"
echo "   python train.py"
echo ""
echo "See QUICKSTART.md for more information!"
echo ""
