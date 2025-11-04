import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMPitchModel(nn.Module):
    """
    CNN + BiLSTM model for converting mel-spectrogram to MIDI notes
    Input: Mel-spectrogram (time, mel_bins)
    Output: MIDI sequence (time, pitch, velocity, duration)
    """
    
    def __init__(self, mel_bins=128, hidden_size=256, num_layers=2, num_pitches=128):
        super(CNNLSTMPitchModel, self).__init__()
        
        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_cnn = nn.Dropout(0.3)
        
        # Calculate CNN output size
        # After 3 pooling layers: mel_bins / 8
        cnn_output_size = 256 * (mel_bins // 8)
        
        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Output heads
        self.fc_pitch = nn.Linear(hidden_size * 2, num_pitches)
        self.fc_velocity = nn.Linear(hidden_size * 2, 128)
        self.fc_onset = nn.Linear(hidden_size * 2, 1)
        self.fc_offset = nn.Linear(hidden_size * 2, 1)
        
        self.dropout_fc = nn.Dropout(0.4)
        
    def forward(self, x):
        """
        Args:
            x: (batch, time, mel_bins)
        Returns:
            pitch: (batch, time, num_pitches)
            velocity: (batch, time, 128)
            onset: (batch, time, 1)
            offset: (batch, time, 1)
        """
        batch_size, time_steps, mel_bins = x.shape
        
        # Reshape for CNN: (batch, 1, time, mel_bins)
        x = x.unsqueeze(1)
        
        # CNN layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_cnn(x)
        
        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, mel)
        x = x.reshape(batch_size, -1, x.size(2) * x.size(3))
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_fc(lstm_out)
        
        # Output predictions
        pitch = torch.softmax(self.fc_pitch(lstm_out), dim=-1)
        velocity = torch.sigmoid(self.fc_velocity(lstm_out)) * 127
        onset = torch.sigmoid(self.fc_onset(lstm_out))
        offset = torch.sigmoid(self.fc_offset(lstm_out))
        
        return pitch, velocity, onset, offset


class TransformerContextModel(nn.Module):
    """
    Transformer model for genre/rhythm adaptation and contextual MIDI generation
    """
    
    def __init__(self, input_dim=256, num_heads=8, num_layers=6, hidden_dim=512, 
                 num_genres=10, num_instruments=20):
        super(TransformerContextModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Genre and instrument embeddings
        self.genre_embedding = nn.Embedding(num_genres, hidden_dim)
        self.instrument_embedding = nn.Embedding(num_instruments, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.note_head = nn.Linear(hidden_dim, 128)
        self.timing_head = nn.Linear(hidden_dim, 32)  # Quantization levels
        self.dynamics_head = nn.Linear(hidden_dim, 128)
        
    def forward(self, x, genre_id, instrument_id):
        """
        Args:
            x: (batch, sequence_length, input_dim)
            genre_id: (batch,)
            instrument_id: (batch,)
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x)
        
        # Add genre and instrument context
        genre_emb = self.genre_embedding(genre_id).unsqueeze(1)
        instrument_emb = self.instrument_embedding(instrument_id).unsqueeze(1)
        
        # Concatenate context tokens
        x = torch.cat([genre_emb, instrument_emb, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Remove context tokens
        x = x[:, 2:, :]
        
        # Generate outputs
        notes = torch.sigmoid(self.note_head(x))
        timing = torch.softmax(self.timing_head(x), dim=-1)
        dynamics = torch.sigmoid(self.dynamics_head(x)) * 127
        
        return notes, timing, dynamics


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


if __name__ == "__main__":
    # Test models
    print("Testing CNN-LSTM Pitch Model...")
    model1 = CNNLSTMPitchModel(mel_bins=128, hidden_size=256)
    test_input = torch.randn(2, 100, 128)  # batch=2, time=100, mel_bins=128
    pitch, velocity, onset, offset = model1(test_input)
    print(f"Pitch shape: {pitch.shape}")
    print(f"Velocity shape: {velocity.shape}")
    print(f"Onset shape: {onset.shape}")
    print(f"Offset shape: {offset.shape}")
    
    print("\nTesting Transformer Context Model...")
    model2 = TransformerContextModel()
    test_input2 = torch.randn(2, 50, 256)
    genre_ids = torch.tensor([0, 1])
    instrument_ids = torch.tensor([5, 10])
    notes, timing, dynamics = model2(test_input2, genre_ids, instrument_ids)
    print(f"Notes shape: {notes.shape}")
    print(f"Timing shape: {timing.shape}")
    print(f"Dynamics shape: {dynamics.shape}")
