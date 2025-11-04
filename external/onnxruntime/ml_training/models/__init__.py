"""
ML Models Package
"""

from .pitch_melody_model import CNNLSTMPitchModel, TransformerContextModel
from .timbre_encoder import TimbreAutoencoder, ContrastiveLoss, TimbreMapper
from .rhythm_gan import MIDIGenerator, MIDIDiscriminator, MultiTrackMIDIGAN
from .continuation_model import RealtimeMIDIContinuation, GRUContinuation

__all__ = [
    'CNNLSTMPitchModel',
    'TransformerContextModel',
    'TimbreAutoencoder',
    'ContrastiveLoss',
    'TimbreMapper',
    'MIDIGenerator',
    'MIDIDiscriminator',
    'MultiTrackMIDIGAN',
    'RealtimeMIDIContinuation',
    'GRUContinuation'
]
