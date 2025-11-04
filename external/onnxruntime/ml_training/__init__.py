"""
Vocal MIDI Generator - ML Training Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .models import (
    pitch_melody_model,
    timbre_encoder,
    rhythm_gan,
    continuation_model
)

__all__ = [
    'pitch_melody_model',
    'timbre_encoder',
    'rhythm_gan',
    'continuation_model'
]
