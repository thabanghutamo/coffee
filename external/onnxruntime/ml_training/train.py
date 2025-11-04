import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models.pitch_melody_model import CNNLSTMPitchModel, TransformerContextModel
from models.timbre_encoder import TimbreAutoencoder, ContrastiveLoss
from models.rhythm_gan import MIDIGenerator, MIDIDiscriminator, MultiTrackMIDIGAN
from models.continuation_model import RealtimeMIDIContinuation
import os


class MIDIDataset(Dataset):
    """
    Dataset for loading MIDI and audio pairs
    Supports: Lakh MIDI, MAESTRO, custom loops
    """
    
    def __init__(self, data_dir, audio_dir=None, split='train'):
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.split = split
        self.samples = []
        
        # Load dataset indices
        self.load_samples()
    
    def load_samples(self):
        # TODO: Implement actual data loading
        # For now, create dummy samples
        self.samples = list(range(1000))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # TODO: Load actual MIDI and audio
        # Return: mel_spectrogram, midi_notes, genre_label, instrument_label
        
        mel_spec = torch.randn(100, 128)  # (time, mel_bins)
        midi_pitch = torch.randn(100, 128)
        midi_velocity = torch.rand(100, 128) * 127
        onset = torch.rand(100)
        offset = torch.rand(100)
        genre_id = torch.randint(0, 10, (1,)).item()
        instrument_id = torch.randint(0, 20, (1,)).item()
        
        return {
            'mel_spec': mel_spec,
            'pitch': midi_pitch,
            'velocity': midi_velocity,
            'onset': onset,
            'offset': offset,
            'genre_id': genre_id,
            'instrument_id': instrument_id
        }


def train_pitch_model(data_loader, num_epochs=100, device='cuda'):
    """
    Train CNN-LSTM pitch detection model
    """
    print("Training Pitch Model...")
    
    model = CNNLSTMPitchModel(mel_bins=128, hidden_size=256).to(device)
    
    criterion_pitch = nn.BCEWithLogitsLoss()  # Use BCE for multi-label classification
    criterion_velocity = nn.MSELoss()
    criterion_onset = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in data_loader:
            mel_spec = batch['mel_spec'].to(device)
            target_pitch = batch['pitch'].to(device)
            target_velocity = batch['velocity'].to(device)
            target_onset = batch['onset'].to(device)
            target_offset = batch['offset'].to(device)
            
            optimizer.zero_grad()
            
            pitch, velocity, onset, offset = model(mel_spec)
            
            # Calculate losses
            # For pitch: use sigmoid activation with BCEWithLogitsLoss
            loss_pitch = criterion_pitch(pitch, target_pitch)
            loss_velocity = criterion_velocity(velocity, target_velocity)
            loss_onset = criterion_onset(onset.squeeze(), target_onset)
            loss_offset = criterion_onset(offset.squeeze(), target_offset)
            
            loss = loss_pitch + loss_velocity + loss_onset + loss_offset
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/pitch_model_epoch_{epoch+1}.pth')
    
    return model


def train_gan(data_loader, num_epochs=200, device='cuda'):
    """
    Train GAN for rhythm and pattern generation
    """
    print("Training MIDI GAN...")
    
    multitrack_gan = MultiTrackMIDIGAN(latent_dim=100, genre_dim=10, condition_dim=64)
    
    generators = multitrack_gan.get_generators()
    discriminators = multitrack_gan.get_discriminators()
    
    # Move to device
    for gen in generators:
        gen.to(device)
    for disc in discriminators:
        disc.to(device)
    
    # Optimizers
    optimizers_G = [optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999)) for gen in generators]
    optimizers_D = [optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999)) for disc in discriminators]
    
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            batch_size = batch['mel_spec'].size(0)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Genre and condition
            genre_emb = torch.randn(batch_size, 10).to(device)
            condition = torch.randn(batch_size, 64).to(device)
            noise = torch.randn(batch_size, 100).to(device)
            
            # Train each generator/discriminator pair
            for i, (gen, disc, opt_G, opt_D) in enumerate(zip(generators, discriminators, optimizers_G, optimizers_D)):
                # Train Discriminator
                opt_D.zero_grad()
                
                real_midi = batch['pitch'].to(device)  # Use actual MIDI
                real_output = disc(real_midi, genre_emb)
                loss_real = criterion(real_output, real_labels)
                
                fake_midi = gen(noise, genre_emb, condition)
                fake_output = disc(fake_midi.detach(), genre_emb)
                loss_fake = criterion(fake_output, fake_labels)
                
                loss_D = loss_real + loss_fake
                loss_D.backward()
                opt_D.step()
                
                # Train Generator
                opt_G.zero_grad()
                
                fake_output = disc(fake_midi, genre_emb)
                loss_G = criterion(fake_output, real_labels)
                
                loss_G.backward()
                opt_G.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")
    
    return multitrack_gan


def export_to_onnx(model, model_name, input_shape, output_path):
    """
    Export trained PyTorch model to ONNX format for C++ inference
    """
    print(f"Exporting {model_name} to ONNX...")
    
    model.eval()
    
    # Create dummy input
    if model_name == "pitch_model":
        dummy_input = torch.randn(1, *input_shape)
    elif model_name == "context_model":
        dummy_input = (torch.randn(1, 50, 256), torch.tensor([0]), torch.tensor([5]))
    else:
        dummy_input = torch.randn(1, *input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Load data
    train_dataset = MIDIDataset('datasets/train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train models
    print("\n=== Training Pitch Model ===")
    pitch_model = train_pitch_model(train_loader, num_epochs=5, device=device)
    
    print("\n=== Training GAN ===")
    gan_models = train_gan(train_loader, num_epochs=5, device=device)
    
    # Export to ONNX
    print("\n=== Exporting Models ===")
    export_to_onnx(pitch_model, "pitch_model", (100, 128), "../models/pitch_model.onnx")
    
    print("\nTraining complete!")
