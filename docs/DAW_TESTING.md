# DAW Testing & Optimization Guide

Complete guide for testing and optimizing Vocal MIDI Generator in professional DAWs.

## Table of Contents

1. [Pre-Testing Setup](#pre-testing-setup)
2. [DAW-Specific Testing](#daw-specific-testing)
3. [Performance Benchmarking](#performance-benchmarking)
4. [Latency Optimization](#latency-optimization)
5. [Troubleshooting](#troubleshooting)

---

## Pre-Testing Setup

### Build the Plugin

```bash
# Configure with optimizations
cd /workspaces/coffee
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build -j$(nproc)
```

### Install Plugin Files

**macOS:**
```bash
# VST3
cp -r build/VocalMIDI_artefacts/Release/VST3/VocalMIDI.vst3 ~/Library/Audio/Plug-Ins/VST3/

# AU
cp -r build/VocalMIDI_artefacts/Release/AU/VocalMIDI.component ~/Library/Audio/Plug-Ins/Components/
```

**Windows:**
```bash
# VST3
copy build\VocalMIDI_artefacts\Release\VST3\VocalMIDI.vst3 "C:\Program Files\Common Files\VST3\"
```

**Linux:**
```bash
# VST3
cp -r build/VocalMIDI_artefacts/Release/VST3/VocalMIDI.vst3 ~/.vst3/
```

### Install ML Models

```bash
# Create models directory next to plugin
mkdir -p build/VocalMIDI_artefacts/Release/Standalone/models/

# Copy trained ONNX models
cp ml_training/exported_models/*.onnx build/VocalMIDI_artefacts/Release/Standalone/models/
```

---

## DAW-Specific Testing

### Ableton Live (macOS/Windows)

**Setup:**
1. Open Ableton Live
2. Create new MIDI track
3. Drag VocalMIDI plugin to track (as MIDI Effect)
4. Create audio track for input
5. Route audio track output to MIDI track sidechain

**Test Cases:**

#### TC-1: Basic Vocal to MIDI
```
1. Record yourself humming a simple melody (C-D-E-F-G)
2. Play back and observe piano roll
3. Verify: Notes match pitch within ±50 cents
4. Verify: Timing is quantized correctly
```

#### TC-2: Real-time Performance
```
1. Set buffer size to 128 samples
2. Enable Input Monitoring
3. Sing/hum in real-time
4. Verify: Latency < 15ms (use Ableton's latency compensation)
5. Verify: No audio dropouts or glitches
```

#### TC-3: Genre Switching
```
1. Load plugin
2. Set Genre to "Trap"
3. Beatbox a pattern
4. Verify: Drums, bass, hi-hats generated
5. Switch to "Pop"
6. Verify: Chord progressions change style
```

#### TC-4: Multi-track Generation
```
1. Sing melody
2. Enable all 4 tracks (Drums, Bass, Chords, Melody)
3. Verify: Each track has distinct MIDI
4. Solo each track and listen
5. Verify: Tracks are musically coherent
```

**Performance Metrics:**
```
Expected:
- Latency: 5-10ms
- CPU: 15-25%
- RAM: 200-400MB
```

### Logic Pro (macOS)

**Setup:**
1. Create Software Instrument track
2. Add VocalMIDI as MIDI FX
3. Add audio track for input with Send to MIDI track

**Test Cases:**

#### TC-5: AU Plugin Validation
```
1. Open Logic Pro
2. Scan plugins (should detect VocalMIDI.component)
3. Load plugin
4. Verify: UI renders correctly
5. Verify: All controls respond
```

#### TC-6: Smart Tempo Integration
```
1. Enable Smart Tempo
2. Sing along to project tempo
3. Verify: MIDI generation follows project BPM
4. Change tempo during playback
5. Verify: Plugin adapts to new tempo
```

### FL Studio (Windows)

**Setup:**
1. Add VocalMIDI to MIDI Out channel
2. Route microphone to sidechain input

**Test Cases:**

#### TC-7: Piano Roll Integration
```
1. Generate MIDI from vocal
2. Open FL Piano Roll
3. Verify: Notes are editable
4. Drag notes, change velocity
5. Verify: Changes persist
```

#### TC-8: Pattern-based Workflow
```
1. Create 4-bar pattern
2. Beatbox drums for 4 bars
3. Enable Loop
4. Verify: Pattern repeats correctly
5. Verify: Timing is locked to grid
```

### Reaper (Cross-platform)

**Setup:**
1. Add VocalMIDI as instrument plugin
2. Configure sidechain routing

**Test Cases:**

#### TC-9: FX Chain Testing
```
1. Load VocalMIDI
2. Add reverb/delay after plugin
3. Verify: Audio passes through
4. Verify: MIDI not affected by audio FX
```

---

## Performance Benchmarking

### Latency Measurement Tool

Use the built-in performance profiler:

```cpp
// In PluginProcessor.cpp
#include "performance/PerformanceProfiler.h"

void processBlock(AudioBuffer<float>& buffer, MidiBuffer& midi)
{
    PROFILE_SCOPE(profiler, "Total_Processing");
    
    {
        PROFILE_SCOPE(profiler, "Feature_Extraction");
        // ... feature extraction code
    }
    
    {
        PROFILE_SCOPE(profiler, "ML_Inference");
        // ... ML inference code
    }
    
    // Generate report every 5 seconds
    if (shouldGenerateReport())
    {
        profiler.saveReportToFile(File("~/Desktop/vocal_midi_profile.txt"));
    }
}
```

### Run Benchmark Suite

```bash
# Build with profiling enabled
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_PROFILING=ON
cmake --build build -j$(nproc)

# Run standalone and load test file
./build/VocalMIDI_artefacts/Release/Standalone/VocalMIDI

# Check profiling output
cat ~/Desktop/vocal_midi_profile.txt
```

**Expected Output:**
```
=== Performance Profiling Report ===

CPU Usage:
  Average: 18.5%
  Peak:    32.1%

Audio Buffer:
  Underruns: 0

Timing Statistics (microseconds):

  Feature_Extraction:
    Count:  1000
    Mean:   850 µs (0.85 ms)
    Min:    720 µs
    Max:    1200 µs
    StdDev: 120 µs

  ML_Inference:
    Count:  1000
    Mean:   4500 µs (4.5 ms)
    Min:    3800 µs
    Max:    6200 µs
    StdDev: 450 µs

  Total_Processing:
    Count:  1000
    Mean:   5800 µs (5.8 ms)
    Min:    5100 µs
    Max:    7900 µs
    StdDev: 520 µs
```

### Stress Testing

```bash
# Multi-instance test
# Load 8 instances of VocalMIDI in DAW
# Play all simultaneously
# Monitor:
#   - Total CPU usage (should stay < 60%)
#   - Audio dropouts (should be 0)
#   - UI responsiveness (should remain smooth)
```

---

## Latency Optimization

### Target: < 10ms End-to-End

#### 1. Optimize Buffer Size

```cpp
// In PluginProcessor.cpp
void prepareToPlay(double sampleRate, int samplesPerBlock)
{
    // Use smaller internal buffers for lower latency
    internalBufferSize = std::min(samplesPerBlock, 128);
    
    // Configure feature extraction for minimal latency
    featureExtraction.setHopSize(64); // 1.45ms at 44.1kHz
    featureExtraction.setWindowSize(512); // 11.6ms at 44.1kHz
}
```

#### 2. Enable Hardware Acceleration

```cpp
// In ModelInference.cpp
sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
sessionOptions.SetIntraOpNumThreads(2);
sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
OrtCUDAProviderOptions cuda_options;
cuda_options.device_id = 0;
cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
#endif
```

#### 3. Use Lock-Free Communication

The `AudioMLBridge` is already lock-free. Verify it's used:

```cpp
// In PluginProcessor::processBlock
AudioDataPacket packet;
packet.melSpectrogram = currentMelSpec;
packet.pitchInfo = currentPitchInfo;
packet.rhythmInfo = currentRhythmInfo;
packet.samplePosition = currentSamplePosition;

// Non-blocking send
if (!audioMLBridge.sendAudioData(std::move(packet)))
{
    // Queue full - skip this frame
    DBG("Warning: Audio queue full, skipping frame");
}

// Non-blocking receive
MIDIResultPacket result;
while (audioMLBridge.receiveMIDIResult(result))
{
    // Add MIDI to buffer
    addMIDIToBuffer(result.midiData, midiMessages);
}
```

#### 4. Profile and Identify Bottlenecks

```bash
# Use Instruments (macOS)
instruments -t "Time Profiler" -D profile.trace build/VocalMIDI_artefacts/Release/Standalone/VocalMIDI

# Use perf (Linux)
perf record -g ./build/VocalMIDI_artefacts/Release/Standalone/VocalMIDI
perf report

# Use Visual Studio Profiler (Windows)
# Open solution, Debug → Performance Profiler → CPU Usage
```

---

## Troubleshooting

### Issue: High Latency (> 20ms)

**Diagnosis:**
```bash
# Check profiling report
cat ~/Desktop/vocal_midi_profile.txt | grep "Mean:"

# If ML_Inference > 10ms:
#   - Use smaller model or quantized ONNX
#   - Enable GPU acceleration
#   - Reduce mel-spectrogram size

# If Feature_Extraction > 2ms:
#   - Use smaller FFT size
#   - Reduce hop size
#   - Optimize mel filterbank
```

**Solutions:**
1. Reduce model complexity
2. Enable CUDA/CoreML
3. Increase DAW buffer size (trade-off: more latency, but stable)

### Issue: Audio Dropouts

**Diagnosis:**
```
1. Check buffer underrun count in profiler
2. Monitor CPU spikes
3. Check lock contention in ThreadAnalyzer
```

**Solutions:**
```cpp
// Increase queue sizes
static constexpr size_t AUDIO_QUEUE_SIZE = 32;  // Was 16
static constexpr size_t MIDI_QUEUE_SIZE = 64;   // Was 32

// Reduce ML inference frequency
if (samplesSinceLastInference > 2048) // Process every 46ms instead of 23ms
{
    runMLInference();
}
```

### Issue: Incorrect Pitch Detection

**Diagnosis:**
1. Record vocals with spectrum analyzer
2. Compare detected pitch with ground truth
3. Check confidence values

**Solutions:**
- Retrain pitch model with more data
- Adjust YIN algorithm threshold
- Use median filtering for stability

### Issue: Plugin Not Detected in DAW

**macOS:**
```bash
# Check plugin signature
codesign -dv ~/Library/Audio/Plug-Ins/VST3/VocalMIDI.vst3

# Re-sign if needed
codesign -s - --force ~/Library/Audio/Plug-Ins/VST3/VocalMIDI.vst3

# Clear AU cache
killall -9 AudioComponentRegistrar
```

**Windows:**
```bash
# Check plugin architecture
dumpbin /headers VocalMIDI.vst3 | findstr "machine"

# Should match DAW (x64 for 64-bit DAW)
```

---

## Automated Testing

### Create Test Suite

```bash
# ml_training/tests/test_plugin_integration.py
import subprocess
import time

def test_latency():
    # Run plugin with test audio
    result = subprocess.run([
        './build/VocalMIDI_artefacts/Release/Standalone/VocalMIDI',
        '--test-mode',
        '--input', 'test_audio/sine_440hz.wav',
        '--output', 'test_output/midi_result.mid'
    ])
    
    # Check latency from profile report
    with open('profile.txt') as f:
        for line in f:
            if 'Total_Processing' in line and 'Mean:' in line:
                latency_us = float(line.split()[3])
                assert latency_us < 10000, f"Latency too high: {latency_us/1000}ms"

def test_pitch_accuracy():
    # Generate test MIDI from known sine wave
    # Compare with expected output
    pass

if __name__ == '__main__':
    test_latency()
    test_pitch_accuracy()
    print("✓ All tests passed!")
```

---

## Performance Targets

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| End-to-End Latency | < 10ms | 10-20ms | > 20ms |
| CPU Usage (single instance) | < 20% | 20-40% | > 40% |
| RAM Usage | < 300MB | 300-500MB | > 500MB |
| Pitch Accuracy | > 95% | 85-95% | < 85% |
| Audio Dropouts (10min session) | 0 | 1-3 | > 3 |
| UI Frame Rate | 60fps | 30-60fps | < 30fps |

---

## Next Steps After Testing

1. **Collect user feedback** from beta testers
2. **Optimize hot paths** identified by profiler
3. **Train models with diverse datasets** for better accuracy
4. **Add preset system** for common use cases
5. **Implement auto-tuning** for optimal latency/quality trade-off

---

**Questions? Open an issue on GitHub or check the documentation.**
