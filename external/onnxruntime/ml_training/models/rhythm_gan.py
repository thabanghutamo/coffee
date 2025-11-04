import torch
import torch.nn as nn
import torch.nn.functional as F

class MIDIGenerator(nn.Module):
    """
    GAN Generator for creating rhythmic MIDI patterns
    """
    
    def __init__(self, latent_dim=100, genre_dim=10, condition_dim=64, output_dim=128):
        super(MIDIGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Concatenate: noise + genre + user rhythm features
        input_dim = latent_dim + genre_dim + condition_dim
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, output_dim * 16)  # 16 time steps
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, noise, genre_embedding, condition):
        """
        Args:
            noise: (batch, latent_dim)
            genre_embedding: (batch, genre_dim)
            condition: (batch, condition_dim) - user rhythm features
        Returns:
            midi_pattern: (batch, 16, 128) - 16 time steps, 128 MIDI notes
        """
        x = torch.cat([noise, genre_embedding, condition], dim=1)
        
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x))
        
        # Reshape to (batch, time_steps, midi_notes)
        x = x.view(-1, 16, 128)
        
        return x


class MIDIDiscriminator(nn.Module):
    """
    GAN Discriminator for evaluating MIDI pattern quality
    """
    
    def __init__(self, input_dim=128, genre_dim=10):
        super(MIDIDiscriminator, self).__init__()
        
        # Process MIDI sequence
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        
        # Flatten and classify
        # After 2 pooling: 16 -> 4 time steps
        self.fc1 = nn.Linear(256 * 4 + genre_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, midi_pattern, genre_embedding):
        """
        Args:
            midi_pattern: (batch, time_steps, midi_notes)
            genre_embedding: (batch, genre_dim)
        Returns:
            validity: (batch, 1) - probability that MIDI is real
        """
        # Transpose for conv1d: (batch, midi_notes, time_steps)
        x = midi_pattern.transpose(1, 2)
        
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Concatenate genre
        x = torch.cat([x, genre_embedding], dim=1)
        
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x


class MultiTrackMIDIGAN:
    """
    Wrapper for generating multi-track MIDI (drums, bass, chords, melody)
    """
    
    def __init__(self, latent_dim=100, genre_dim=10, condition_dim=64):
        self.drum_generator = MIDIGenerator(latent_dim, genre_dim, condition_dim, output_dim=128)
        self.bass_generator = MIDIGenerator(latent_dim, genre_dim, condition_dim, output_dim=128)
        self.chord_generator = MIDIGenerator(latent_dim, genre_dim, condition_dim, output_dim=128)
        self.melody_generator = MIDIGenerator(latent_dim, genre_dim, condition_dim, output_dim=128)
        
        self.drum_discriminator = MIDIDiscriminator(input_dim=128, genre_dim=genre_dim)
        self.bass_discriminator = MIDIDiscriminator(input_dim=128, genre_dim=genre_dim)
        self.chord_discriminator = MIDIDiscriminator(input_dim=128, genre_dim=genre_dim)
        self.melody_discriminator = MIDIDiscriminator(input_dim=128, genre_dim=genre_dim)
        
    def generate_multitrack(self, noise, genre_embedding, condition):
        """
        Generate all tracks simultaneously
        """
        drums = self.drum_generator(noise, genre_embedding, condition)
        bass = self.bass_generator(noise, genre_embedding, condition)
        chords = self.chord_generator(noise, genre_embedding, condition)
        melody = self.melody_generator(noise, genre_embedding, condition)
        
        return {
            'drums': drums,
            'bass': bass,
            'chords': chords,
            'melody': melody
        }
    
    def get_generators(self):
        return [self.drum_generator, self.bass_generator, 
                self.chord_generator, self.melody_generator]
    
    def get_discriminators(self):
        return [self.drum_discriminator, self.bass_discriminator,
                self.chord_discriminator, self.melody_discriminator]


if __name__ == "__main__":
    # Test GAN models
    print("Testing MIDI Generator...")
    generator = MIDIGenerator(latent_dim=100, genre_dim=10, condition_dim=64)
    noise = torch.randn(4, 100)
    genre = torch.randn(4, 10)
    condition = torch.randn(4, 64)
    generated_midi = generator(noise, genre, condition)
    print(f"Generated MIDI shape: {generated_midi.shape}")
    
    print("\nTesting MIDI Discriminator...")
    discriminator = MIDIDiscriminator(input_dim=128, genre_dim=10)
    validity = discriminator(generated_midi, genre)
    print(f"Validity shape: {validity.shape}")
    print(f"Validity values: {validity[:3].squeeze()}")
    
    print("\nTesting Multi-Track GAN...")
    multitrack_gan = MultiTrackMIDIGAN()
    tracks = multitrack_gan.generate_multitrack(noise, genre, condition)
    for track_name, track_data in tracks.items():
        print(f"{track_name}: {track_data.shape}")
