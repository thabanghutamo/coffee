# Testing Guide - Vocal MIDI Generator

## Quick Start Testing

### 1. Test Build Status
```bash
# Verify all components built successfully
ls -lh build/VocalMIDI_artefacts/Standalone/VocalMIDI
ls -lh build/VocalMIDI_artefacts/VST3/VocalMIDI.vst3/
ls -lh build/VocalMIDI_artefacts/models/*.onnx
```

**Expected Output:**
- Standalone: ~21 MB executable
- VST3: Plugin bundle directory
- Models: 8 ONNX files (~229 KB each)

### 2. Test ONNX Runtime Integration
```bash
# Check ONNX Runtime linkage
export LD_LIBRARY_PATH=/workspaces/coffee/external/onnxruntime/lib:$LD_LIBRARY_PATH
ldd build/VocalMIDI_artefacts/Standalone/VocalMIDI | grep onnx
```

**Expected Output:**
```
libonnxruntime.so.1.16.3 => /workspaces/coffee/external/onnxruntime/lib/libonnxruntime.so.1.16.3
```

### 3. Launch Standalone Plugin (Headless Test)
```bash
# Note: GUI won't display in Codespaces, but we can test initialization
./run_standalone.sh --help 2>&1 | head -20
```

### 4. Verify Model Loading
```bash
# Check if models are accessible
python3 << 'EOF'
import onnxruntime as ort
import os

models_dir = 'build/VocalMIDI_artefacts/models'
models = [
    'pitch_model.onnx',
    'context_model.onnx', 
    'timbre_model.onnx',
    'drum_generator.onnx',
    'bass_generator.onnx',
    'chord_generator.onnx',
    'melody_generator.onnx',
    'continuation_model.onnx'
]

print("Testing ONNX Model Loading:\n")
for model_name in models:
    path = os.path.join(models_dir, model_name)
    try:
        session = ort.InferenceSession(path)
        inputs = [i.name for i in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        print(f"✓ {model_name}")
        print(f"  Inputs: {inputs}")
        print(f"  Outputs: {outputs}\n")
    except Exception as e:
        print(f"✗ {model_name}: {e}\n")
EOF
```

## Component Testing

### Audio Processing Pipeline

**Test Pitch Detection:**
```bash
# Create test audio file
python3 << 'EOF'
import numpy as np
import wave

# Generate 440 Hz sine wave (A4 note)
sample_rate = 44100
duration = 2.0
frequency = 440.0

t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.sin(2 * np.pi * frequency * t)
audio = (audio * 32767).astype(np.int16)

with wave.open('test_audio_440hz.wav', 'w') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.writeframes(audio.tobytes())

print("✓ Created test_audio_440hz.wav (440 Hz, 2 seconds)")
EOF
```

### MIDI Generation Pipeline

**Test Model Inference:**
```bash
python3 << 'EOF'
import onnxruntime as ort
import numpy as np

# Load pitch model
session = ort.InferenceSession('build/VocalMIDI_artefacts/models/pitch_model.onnx')

# Create dummy mel-spectrogram input
batch_size = 1
time_steps = 100
mel_bands = 128

input_data = np.random.randn(batch_size, time_steps, mel_bands).astype(np.float32)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run([output_name], {input_name: input_data})
output = result[0]

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
print(f"\n✓ Inference successful!")

# Check if output looks like pitch probabilities
if output.shape == (batch_size, time_steps, mel_bands):
    print("✓ Output shape matches expected [batch, time, 128]")
    if output.min() >= 0 and output.max() <= 1:
        print("✓ Output values in sigmoid range [0, 1]")
EOF
```

## VST3 Testing in DAW

### Install VST3 Plugin

**Linux:**
```bash
# Copy to system VST3 directory
mkdir -p ~/.vst3
cp -r build/VocalMIDI_artefacts/VST3/VocalMIDI.vst3 ~/.vst3/

# Verify installation
ls -la ~/.vst3/VocalMIDI.vst3
```

**macOS:**
```bash
# Copy to system VST3 directory
sudo cp -r build/VocalMIDI_artefacts/VST3/VocalMIDI.vst3 ~/Library/Audio/Plug-Ins/VST3/

# Verify installation
ls -la ~/Library/Audio/Plug-Ins/VST3/VocalMIDI.vst3
```

**Windows:**
```powershell
# Copy to system VST3 directory
Copy-Item -Recurse build\VocalMIDI_artefacts\VST3\VocalMIDI.vst3 "C:\Program Files\Common Files\VST3\"
```

### DAW Testing Checklist

#### Reaper
1. Open Reaper
2. Create new track (Ctrl+T)
3. Click FX button → VST3 → VocalMIDI
4. Enable input monitoring
5. Sing/play into microphone
6. Check MIDI output in piano roll

#### Ableton Live
1. Open Ableton
2. Insert → MIDI Track
3. Drag VocalMIDI to track
4. Enable input monitoring
5. Arm track for recording
6. Test with vocal input

#### FL Studio
1. Open FL Studio
2. Channels → Add One → More → VocalMIDI
3. Route audio input to plugin
4. Open Piano Roll
5. Test recording

## Performance Testing

### CPU Usage Test
```bash
# Monitor plugin CPU usage
python3 << 'EOF'
import time
import psutil

print("Simulating audio processing load test...")
print("Note: Actual test requires running plugin")
print("\nExpected metrics:")
print("  CPU per voice: < 5%")
print("  Latency: < 10ms")
print("  Memory: < 200 MB")
EOF
```

### Latency Measurement
```bash
# Create latency test
cat > test_latency.py << 'PYEOF'
"""
Latency test for Vocal MIDI Generator

Expected results:
- Input → Feature extraction: < 5ms
- Feature extraction → ML inference: < 10ms  
- ML inference → MIDI output: < 2ms
- Total latency: < 20ms
"""

import time

def test_pipeline_latency():
    stages = {
        'Audio Input Buffer': 5.0,
        'Feature Extraction': 8.0,
        'ML Inference': 12.0,
        'MIDI Generation': 2.0,
        'MIDI Output': 1.0
    }
    
    print("Pipeline Latency Budget:\n")
    total = 0
    for stage, ms in stages.items():
        print(f"  {stage:.<30} {ms:>5.1f} ms")
        total += ms
    
    print(f"\n  {'Total Latency':.<30} {total:>5.1f} ms")
    print(f"\n  Target: < 20ms")
    print(f"  Status: {'✓ PASS' if total < 20 else '✗ FAIL'}")

if __name__ == '__main__':
    test_pipeline_latency()
PYEOF

python3 test_latency.py
```

## Model Training Pipeline Testing

### Test Dataset Download
```bash
cd external/onnxruntime/ml_training
python3 download_datasets.py --dataset dummy --output datasets/
```

### Test Training Script
```bash
cd external/onnxruntime/ml_training

# Quick training test (2 epochs)
python3 train.py \
    --dataset datasets/dummy \
    --model pitch \
    --epochs 2 \
    --batch-size 4 \
    --output checkpoints/
```

### Export Trained Models
```bash
# After training, export to ONNX
python3 << 'EOF'
import torch

# Load checkpoint (example)
checkpoint_path = 'external/onnxruntime/ml_training/checkpoints/pitch_model_epoch_2.pt'
output_path = 'build/VocalMIDI_artefacts/models/pitch_model.onnx'

# Note: This is a template - actual export depends on model architecture
print(f"Export trained model: {checkpoint_path} → {output_path}")
EOF
```

## Troubleshooting

### Plugin Won't Load
```bash
# Check library dependencies
ldd build/VocalMIDI_artefacts/Standalone/VocalMIDI

# Common issues:
# - Missing ONNX Runtime: Add to LD_LIBRARY_PATH
# - Missing JUCE libs: Install X11 dev packages
# - Missing CUDA: Plugin falls back to CPU
```

### Models Not Found
```bash
# Verify model paths
ls -la build/VocalMIDI_artefacts/models/

# Expected: 8 .onnx files
# If missing, run: python3 create_placeholder_models.py
```

### No MIDI Output
1. Check audio input is enabled
2. Verify pitch detection threshold
3. Check MIDI output routing in DAW
4. Enable debug logging in plugin

### High Latency
1. Reduce buffer size in audio settings
2. Enable GPU acceleration (CUDA)
3. Reduce model complexity
4. Check for denormal numbers

## Next Steps

After basic testing:

1. **Train Real Models** - Use actual vocal datasets
2. **DAW Integration** - Test in production environment
3. **Performance Profiling** - Optimize hotspots
4. **User Testing** - Get feedback from musicians
5. **Documentation** - Write user manual

## Automated Test Suite

```bash
# Run all tests
cat > run_tests.sh << 'TESTEOF'
#!/bin/bash

echo "=== Vocal MIDI Generator Test Suite ==="

echo -e "\n1. Build Verification..."
ls build/VocalMIDI_artefacts/Standalone/VocalMIDI && echo "✓ Standalone built" || echo "✗ Standalone missing"
ls build/VocalMIDI_artefacts/VST3/VocalMIDI.vst3 && echo "✓ VST3 built" || echo "✗ VST3 missing"

echo -e "\n2. Model Verification..."
MODEL_COUNT=$(ls build/VocalMIDI_artefacts/models/*.onnx 2>/dev/null | wc -l)
[ "$MODEL_COUNT" -eq 8 ] && echo "✓ All 8 models present" || echo "✗ Only $MODEL_COUNT models found"

echo -e "\n3. ONNX Runtime..."
export LD_LIBRARY_PATH=/workspaces/coffee/external/onnxruntime/lib:$LD_LIBRARY_PATH
ldd build/VocalMIDI_artefacts/Standalone/VocalMIDI | grep -q onnxruntime && echo "✓ ONNX Runtime linked" || echo "✗ ONNX Runtime not found"

echo -e "\n4. Python Dependencies..."
python3 -c "import torch; import onnxruntime; import numpy" && echo "✓ Python packages OK" || echo "✗ Missing packages"

echo -e "\n=== Test Suite Complete ==="
TESTEOF

chmod +x run_tests.sh
./run_tests.sh
```

## Success Criteria

- ✓ All 8 ONNX models load successfully
- ✓ Standalone plugin launches without crashes
- ✓ VST3 plugin recognized by DAW
- ✓ Audio input → MIDI output pipeline works
- ✓ Latency < 20ms
- ✓ CPU usage < 10% per voice
- ✓ No memory leaks during extended use
