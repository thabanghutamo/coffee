# ML Training Package

This package contains all PyTorch models and training utilities for the Vocal MIDI Generator.

## Modules

- `models/` - PyTorch model architectures
  - `pitch_melody_model.py` - CNN-LSTM pitch detection
  - `timbre_encoder.py` - Autoencoder for instrument timbre
  - `rhythm_gan.py` - GAN for rhythm generation
  - `continuation_model.py` - LSTM for real-time continuation

- `utils/` - Training utilities
  - `data_preprocessing.py` - Audio/MIDI preprocessing

## Quick Start

```python
from models.pitch_melody_model import CNNLSTMPitchModel
from models.rhythm_gan import MultiTrackMIDIGAN

# Load model
pitch_model = CNNLSTMPitchModel()
pitch_model.load_state_dict(torch.load('checkpoints/pitch_model.pth'))

# Generate
gan = MultiTrackMIDIGAN()
tracks = gan.generate_multitrack(noise, genre, condition)
```

## Training

See `train.py` for the complete training pipeline.
