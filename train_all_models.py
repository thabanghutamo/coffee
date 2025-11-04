#!/usr/bin/env python3
"""
Train all 8 models for Vocal MIDI Generator
This script trains models on synthetic data for testing purposes.
For production, replace with real vocal/MIDI datasets.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple

class PitchModel(nn.Module):
    """Detects pitch from mel-spectrogram"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 128)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = torch.sigmoid(self.fc(lstm_out))
        return output

class ContextModel(nn.Module):
    """Adds musical context and velocity dynamics"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 96, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(96, 4, batch_first=True)
        self.fc = nn.Linear(96, 128)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        output = torch.sigmoid(self.fc(attn_out))
        return output

class TimbreModel(nn.Module):
    """Encodes timbre features"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 128)
        
    def forward(self, x):
        # x: [batch, time, 128]
        x = x.transpose(1, 2)  # [batch, 128, time]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)  # [batch, time, 32]
        lstm_out, _ = self.lstm(x)
        output = torch.sigmoid(self.fc(lstm_out))
        return output

class InstrumentGenerator(nn.Module):
    """Generates instrument-specific MIDI patterns"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 128, num_layers=3, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = torch.relu(self.fc1(lstm_out))
        output = torch.sigmoid(self.fc2(x))
        return output

def generate_synthetic_data(num_samples: int = 500, seq_length: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic training data"""
    mel_bands = 128
    
    # Input: mel-spectrograms
    X = torch.randn(num_samples, seq_length, mel_bands)
    
    # Target: pitch probabilities (simulate realistic patterns)
    y = torch.zeros(num_samples, seq_length, 128)
    
    for i in range(num_samples):
        # Generate melodic patterns
        base_note = torch.randint(48, 72, (1,)).item()
        
        for t in range(seq_length):
            # Occasional note changes (musical phrasing)
            if t % 10 == 0:
                base_note += torch.randint(-2, 3, (1,)).item()
                base_note = max(48, min(72, base_note))
            
            # Add some probability to the note and harmonics
            y[i, t, base_note] = 0.9
            if base_note + 12 < 128:  # Octave
                y[i, t, base_note + 12] = 0.3
            if base_note + 7 < 128:  # Fifth
                y[i, t, base_note + 7] = 0.2
    
    return X, y

def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                model_name: str, epochs: int = 10, batch_size: int = 16):
    """Train a model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    
    num_samples = X.shape[0]
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, num_samples, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            status = "✓ Best"
        else:
            status = ""
        
        print(f"  Epoch {epoch+1:2d}/{epochs} - Loss: {avg_loss:.4f} {status}")
    
    return model

def export_to_onnx(model: nn.Module, output_path: Path, seq_length: int = 100):
    """Export model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(1, seq_length, 128)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['mel_spectrogram'],
        output_names=['output'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch', 1: 'time'},
            'output': {0: 'batch', 1: 'time'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    file_size_kb = output_path.stat().st_size / 1024
    print(f"  ✓ Exported to {output_path.name} ({file_size_kb:.1f} KB)")

def main():
    parser = argparse.ArgumentParser(description='Train all Vocal MIDI Generator models')
    parser.add_argument('--samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--output', type=str, default='build/VocalMIDI_artefacts/models',
                        help='Output directory for ONNX models')
    args = parser.parse_args()
    
    print("="*60)
    print("Vocal MIDI Generator - Full Model Training")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Training samples: {args.samples}")
    print(f"  Epochs per model: {args.epochs}")
    print(f"  Output directory: {args.output}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    print(f"\nGenerating synthetic training data...")
    X_train, y_train = generate_synthetic_data(args.samples)
    print(f"✓ Created {args.samples} samples")
    print(f"  Input shape: {X_train.shape}")
    print(f"  Target shape: {y_train.shape}")
    
    # Define all models
    models = {
        'pitch_model.onnx': PitchModel(),
        'context_model.onnx': ContextModel(),
        'timbre_model.onnx': TimbreModel(),
        'drum_generator.onnx': InstrumentGenerator(),
        'bass_generator.onnx': InstrumentGenerator(),
        'chord_generator.onnx': InstrumentGenerator(),
        'melody_generator.onnx': InstrumentGenerator(),
        'continuation_model.onnx': InstrumentGenerator(),
    }
    
    # Train and export each model
    for model_name, model in models.items():
        trained_model = train_model(model, X_train, y_train, model_name, args.epochs)
        output_path = output_dir / model_name
        export_to_onnx(trained_model, output_path)
    
    print(f"\n{'='*60}")
    print("✅ All models trained and exported successfully!")
    print(f"{'='*60}")
    print(f"\nModels saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Test models with: ./run_tests.sh")
    print("2. Launch standalone: ./run_standalone.sh")
    print("3. Test in DAW with VST3 plugin")
    print("4. Replace with real dataset for production")

if __name__ == '__main__':
    main()
