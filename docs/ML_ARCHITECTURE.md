# ML Architecture Documentation

## Overview

The Vocal MIDI Generator uses a multi-stage ML pipeline to convert vocal input into expressive MIDI. Each stage is optimized for real-time performance while maintaining high accuracy.

## Pipeline Flow

```
Audio Input (Microphone)
    ↓
[Feature Extraction] → Mel-Spectrogram (128 bands)
    ↓
[Pitch Detection] → CNN-LSTM Model → MIDI Notes (pitch, velocity, timing)
    ↓
[Rhythm Analysis] → Onset Detection + Tempo Tracking
    ↓
[Genre/Context] → Transformer Model → Contextual Adaptations
    ↓
[Timbre Matching] → Autoencoder → Instrument Selection
    ↓
[Multi-Track Generation] → GANs → Drums, Bass, Chords, Melody
    ↓
[Real-time Continuation] → LSTM/GRU → Next-step Predictions
    ↓
MIDI Output (Multi-track)
```

## Model Details

### 1. CNN-LSTM Pitch Model

**Architecture:**
```
Input: (batch, time, 128 mel_bins)
    ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(128) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(256) → BatchNorm → ReLU → MaxPool
    ↓
Reshape → (batch, time, features)
    ↓
BiLSTM(256 hidden, 2 layers)
    ↓
├─ Pitch Head (128 pitches)
├─ Velocity Head (128 velocities)
├─ Onset Head (binary)
└─ Offset Head (binary)
```

**Training:**
- Loss: Cross-entropy (pitch) + MSE (velocity) + BCE (onset/offset)
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: 100
- Dataset: Lakh MIDI + MAESTRO

**Performance:**
- Latency: <5ms
- Pitch accuracy: 95%+
- Memory: 50MB

### 2. Transformer Context Model

**Architecture:**
```
Input: (batch, seq_len, 256) + genre_id + instrument_id
    ↓
Genre Embedding (10D) + Instrument Embedding (10D)
    ↓
Positional Encoding
    ↓
Transformer Encoder (6 layers, 8 heads)
    ↓
├─ Note Head
├─ Timing Head (quantization)
└─ Dynamics Head
```

**Purpose:**
- Adapt MIDI to genre-specific patterns
- Apply rhythmic swing and feel
- Intelligent note quantization

**Training:**
- Loss: Combined cross-entropy
- Optimizer: AdamW (lr=0.0001, weight decay=0.01)
- Batch size: 16
- Epochs: 100

### 3. Timbre Autoencoder

**Architecture:**
```
Encoder:
    Input (128D) → Linear(256) → ReLU → Linear(128) → ReLU → Latent(64D)

Decoder:
    Latent(64D) → Linear(128) → ReLU → Linear(256) → ReLU → Output(128D)

Contrastive Loss:
    Pulls similar instruments together in latent space
    Pushes different instruments apart
```

**Training:**
- Loss: Reconstruction (MSE) + Contrastive
- Temperature: 0.5
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Epochs: 150

**Instrument Embeddings:**
Precomputed for 20 instruments:
- Piano, Synth, Guitar, Bass, Drums
- Strings, Brass, Woodwinds, Organ, Pad
- Electric Piano, Acoustic Guitar, etc.

### 4. Multi-Track GAN

**Generator Architecture (per track):**
```
Input: Noise(100D) + Genre(10D) + Condition(64D)
    ↓
Linear(256) → BatchNorm → LeakyReLU → Dropout
    ↓
Linear(512) → BatchNorm → LeakyReLU → Dropout
    ↓
Linear(1024) → BatchNorm → LeakyReLU
    ↓
Linear(128 * 16) → Sigmoid
    ↓
Reshape → (batch, 16 timesteps, 128 notes)
```

**Discriminator Architecture:**
```
Input: (batch, 16 timesteps, 128 notes)
    ↓
Conv1D(256, kernel=3) → LeakyReLU → MaxPool
    ↓
Conv1D(512, kernel=3) → LeakyReLU → MaxPool
    ↓
Conv1D(256, kernel=3) → LeakyReLU
    ↓
Flatten + Concat(Genre Embedding)
    ↓
Linear(256) → LeakyReLU → Dropout
    ↓
Linear(64) → LeakyReLU → Dropout
    ↓
Linear(1) → Sigmoid (real/fake)
```

**Training:**
- Loss: Binary cross-entropy
- Optimizer: Adam (lr=0.0002, betas=(0.5, 0.999))
- Batch size: 32
- Epochs: 200
- Separate GANs for: Drums, Bass, Chords, Melody

### 5. LSTM Continuation Model

**Architecture:**
```
Input: (batch, seq_len, 128)
    ↓
LSTM(256 hidden, 2 layers, dropout=0.2)
    ↓
├─ Note Head (128)
├─ Velocity Head (128)
└─ Timing Head (16)
```

**Inference:**
- Auto-regressive prediction
- Context window: Last 8 timesteps
- Predicts next 4 timesteps
- Latency: <3ms per prediction

## Optimization Techniques

### For Real-time Performance

1. **Model Quantization:**
   - Convert FP32 to FP16 where possible
   - ONNX Runtime optimization level 3
   - Graph optimizations

2. **Batch Size:**
   - Inference batch size = 1 for low latency
   - Use larger batches only for multi-track generation

3. **Memory Management:**
   - Preallocate buffers
   - Reuse tensors
   - Stream processing for long inputs

4. **Threading:**
   - Audio thread: Feature extraction only
   - Background thread: ML inference
   - Callback communication via lock-free queue

## ONNX Export

All models are exported to ONNX format for C++ inference:

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}}
)
```

## Dataset Requirements

### Audio-MIDI Pairs
- **Format**: WAV/MP3 + MIDI
- **Sample rate**: 22050 Hz or 44100 Hz
- **Duration**: 5-60 seconds per sample
- **Minimum**: 10,000 samples for basic training
- **Recommended**: 100,000+ samples for production

### Labeling
- Genre classification
- Instrument type
- BPM/tempo
- Key signature (optional)
- Mood/feel (optional)

## Training Pipeline

1. **Data Preprocessing**
   - Extract mel-spectrograms
   - Convert MIDI to piano roll
   - Align audio and MIDI
   - Augmentation (pitch shift, time stretch)

2. **Training**
   - Stage 1: Pitch model (2-3 days on V100)
   - Stage 2: Context model (1-2 days)
   - Stage 3: Timbre encoder (1 day)
   - Stage 4: GANs (3-5 days)
   - Stage 5: Continuation model (1 day)

3. **Validation**
   - Pitch accuracy on test set
   - Perceptual evaluation (MOS)
   - Latency benchmarks
   - Memory usage

4. **Export & Integration**
   - Export to ONNX
   - Test in C++ plugin
   - Performance profiling
   - User testing

## Future Improvements

- [ ] Attention mechanisms for better long-range dependencies
- [ ] Transformer-based GANs
- [ ] Few-shot learning for new genres
- [ ] Reinforcement learning from user feedback
- [ ] Compressed models for mobile deployment
