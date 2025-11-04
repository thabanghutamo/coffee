# Quick Start Guide

Get the Vocal MIDI Generator running in 15 minutes!

## Prerequisites Check

Before starting, verify you have:

```bash
# Check CMake
cmake --version  # Need 3.22+

# Check compiler
g++ --version    # Need GCC 11+ (Linux)
# OR
clang --version  # Need Clang 14+ (macOS)

# Check Python
python --version # Need 3.9+

# Check Git
git --version
```

---

## 5-Step Setup

### Step 1: Clone & Setup JUCE (2 minutes)

```bash
cd /workspaces/coffee

# Get JUCE framework
git clone --depth 1 --branch 7.0.9 \
    https://github.com/juce-framework/JUCE.git external/JUCE

# Verify
ls external/JUCE/modules
# Should show: juce_audio_basics, juce_audio_devices, etc.
```

### Step 2: Install ONNX Runtime (3 minutes)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y libonnxruntime-dev
```

**macOS:**
```bash
brew install onnxruntime
```

**Windows:**
Download from: https://github.com/microsoft/onnxruntime/releases
Extract to: `C:\Program Files\onnxruntime`

### Step 3: Build the Plugin (5 minutes)

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release -j8
```

**Output:** Compiled plugin in `build/VocalMIDI_artefacts/`

### Step 4: Install Plugin (1 minute)

**Linux:**
```bash
sudo cmake --install .
# Or manually copy to ~/.vst3/
```

**macOS:**
```bash
sudo cmake --install .
# Plugins → /Library/Audio/Plug-Ins/VST3/
```

**Windows:**
```bash
# Copy from build/VocalMIDI_artefacts/Release/VST3/
# To: C:\Program Files\Common Files\VST3\
```

### Step 5: Test in DAW (2 minutes)

1. Open your DAW (Ableton, Logic, FL Studio, etc.)
2. Rescan plugins
3. Insert "VocalMIDI" on a MIDI track
4. Route your microphone
5. Hit record and sing! 🎤

---

## ML Training (Optional)

Want to train your own models?

### Quick Training Setup

```bash
cd ml_training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test models (no training)
python -c "from models import CNNLSTMPitchModel; print('Models loaded!')"
```

### Full Training (requires dataset)

```bash
# Download dataset (example: Lakh MIDI)
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzf lmd_full.tar.gz -C datasets/

# Preprocess
python utils/data_preprocessing.py

# Train (takes hours/days on GPU)
python train.py

# Export to ONNX
# Models saved to ../models/*.onnx
```

---

## Troubleshooting

### "JUCE not found"
```bash
# Make sure JUCE is in the right place
ls external/JUCE/CMakeLists.txt
# Should exist
```

### "ONNX Runtime not found"
```bash
# Linux: Check library path
ldconfig -p | grep onnxruntime

# macOS: Reinstall
brew reinstall onnxruntime

# Windows: Set environment variable
# ONNXRUNTIME_DIR=C:\Program Files\onnxruntime
```

### "Linking errors"
```bash
# Clean rebuild
rm -rf build
mkdir build && cd build
cmake .. && cmake --build .
```

### "Plugin doesn't show in DAW"
- **Check plugin format**: VST3 vs AU vs VST2
- **Scan paths**: Make sure DAW scans correct directory
- **Permissions**: Some DAWs need admin/root to install
- **Compatibility**: Check DAW supports VST3/AU

---

## Development Workflow

### Edit → Build → Test Loop

```bash
# 1. Make changes in src/
vim src/PluginProcessor.cpp

# 2. Rebuild (fast, only changed files)
cd build
cmake --build . -j8

# 3. Test standalone (no DAW needed)
./VocalMIDI_artefacts/Release/Standalone/VocalMIDI

# Or load in DAW
```

### Debugging

**VSCode:**
1. Install CMake Tools extension
2. Open `/workspaces/coffee`
3. `Ctrl+Shift+P` → "CMake: Debug"

**Command line:**
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
gdb ./VocalMIDI_artefacts/Debug/Standalone/VocalMIDI
```

---

## Project Structure at a Glance

```
coffee/
├── src/              ← C++ plugin code (edit here)
│   ├── audio/       ← Audio processing
│   ├── ml/          ← ML inference
│   ├── midi/        ← MIDI generation
│   └── ui/          ← User interface
│
├── ml_training/     ← Python ML training
│   ├── models/      ← PyTorch models
│   └── utils/       ← Preprocessing
│
├── build/           ← Build output (don't edit)
├── models/          ← Trained ONNX models
└── docs/            ← Documentation
```

---

## Common Tasks

### Add a New Genre

1. Edit `src/ml/GenreClassifier.cpp`:
```cpp
std::vector<std::string> getSupportedGenres() const
{
    return { "Trap", "Pop", "YourNewGenre", ... };
}
```

2. Edit `src/ui/ControlPanel.cpp`:
```cpp
genreSelector.addItem("YourNewGenre", 7);
```

3. Rebuild and test!

### Change BPM Range

Edit `src/ui/ControlPanel.cpp`:
```cpp
bpmSlider.setRange(60.0, 200.0, 1.0);  // min, max, step
```

### Adjust Pitch Detection Sensitivity

Edit `src/audio/PitchDetector.cpp`:
```cpp
const float threshold = 0.15f;  // Lower = more sensitive
```

---

## Performance Tips

### Optimize Build
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

### Reduce Latency
Edit `src/PluginProcessor.cpp`:
```cpp
blockSize = 256;  // Smaller = lower latency (but more CPU)
```

### Memory Usage
Models are loaded on startup. To reduce memory:
- Use FP16 instead of FP32 (ONNX export)
- Quantize models
- Load models on-demand

---

## Next Steps

✅ Plugin is built and running!

**Now you can:**

1. **Start using it**
   - Record vocals → Get MIDI
   - Edit in piano roll
   - Export to DAW

2. **Train your models**
   - Follow `ml_training/README.md`
   - Use your own datasets
   - Fine-tune for your style

3. **Customize**
   - Add new genres
   - Modify UI colors
   - Add new instruments
   - Create presets

4. **Contribute**
   - Report issues
   - Submit PRs
   - Share presets
   - Help with docs

---

## Getting Help

- 📖 **Full docs**: `README.md`, `docs/BUILD.md`
- 🔧 **ML guide**: `docs/ML_ARCHITECTURE.md`
- 💬 **Issues**: GitHub Issues
- 🎵 **Community**: Discussions

---

## Quick Command Reference

```bash
# Build
cmake .. && cmake --build . -j8

# Clean rebuild
rm -rf build && mkdir build && cd build && cmake .. && cmake --build .

# Install
sudo cmake --install .

# Test standalone
./build/VocalMIDI_artefacts/Release/Standalone/VocalMIDI

# Train ML models
cd ml_training && python train.py

# Run tests
cd build && ctest
```

---

**You're ready to go! Start singing and generating MIDI!** 🎤→🎹
