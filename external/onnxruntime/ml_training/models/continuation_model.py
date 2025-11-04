import torch
import torch.nn as nn

class RealtimeMIDIContinuation(nn.Module):
    """
    Lightweight LSTM for real-time MIDI continuation and prediction
    Predicts next notes, fills, and counter-melodies
    """
    
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, output_dim=128):
        super(RealtimeMIDIContinuation, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output head
        self.fc_notes = nn.Linear(hidden_dim, output_dim)
        self.fc_velocity = nn.Linear(hidden_dim, output_dim)
        self.fc_timing = nn.Linear(hidden_dim, 16)  # Timing offset
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, sequence_length, input_dim)
            hidden: Optional hidden state from previous step
        Returns:
            notes: (batch, sequence_length, output_dim)
            velocity: (batch, sequence_length, output_dim)
            timing: (batch, sequence_length, 16)
            hidden: Updated hidden state
        """
        lstm_out, hidden = self.lstm(x, hidden)
        
        notes = torch.sigmoid(self.fc_notes(lstm_out))
        velocity = torch.sigmoid(self.fc_velocity(lstm_out)) * 127
        timing = torch.softmax(self.fc_timing(lstm_out), dim=-1)
        
        return notes, velocity, timing, hidden
    
    def predict_next(self, current_sequence, num_steps=4, hidden=None):
        """
        Predict next num_steps of MIDI given current sequence
        """
        predictions = []
        current_input = current_sequence
        
        for _ in range(num_steps):
            notes, velocity, timing, hidden = self.forward(current_input, hidden)
            
            # Get last prediction
            last_note = notes[:, -1:, :]
            last_velocity = velocity[:, -1:, :]
            
            predictions.append({
                'notes': last_note,
                'velocity': last_velocity,
                'timing': timing[:, -1:, :]
            })
            
            # Use prediction as next input
            current_input = last_note
        
        return predictions, hidden


class GRUContinuation(nn.Module):
    """
    Alternative lightweight model using GRU (faster than LSTM)
    """
    
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, output_dim=128):
        super(GRUContinuation, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc_notes = nn.Linear(hidden_dim, output_dim)
        self.fc_velocity = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden=None):
        gru_out, hidden = self.gru(x, hidden)
        
        notes = torch.sigmoid(self.fc_notes(gru_out))
        velocity = torch.sigmoid(self.fc_velocity(gru_out)) * 127
        
        return notes, velocity, hidden


if __name__ == "__main__":
    # Test real-time continuation model
    print("Testing LSTM Continuation Model...")
    model = RealtimeMIDIContinuation(input_dim=128, hidden_dim=256, num_layers=2)
    
    # Simulate current MIDI sequence
    current_sequence = torch.randn(1, 8, 128)  # batch=1, 8 time steps
    
    notes, velocity, timing, hidden = model(current_sequence)
    print(f"Notes shape: {notes.shape}")
    print(f"Velocity shape: {velocity.shape}")
    print(f"Timing shape: {timing.shape}")
    
    # Test prediction
    print("\nTesting next-step prediction...")
    predictions, hidden = model.predict_next(current_sequence, num_steps=4)
    print(f"Generated {len(predictions)} future steps")
    
    print("\nTesting GRU Continuation Model...")
    gru_model = GRUContinuation()
    notes_gru, velocity_gru, hidden_gru = gru_model(current_sequence)
    print(f"GRU Notes shape: {notes_gru.shape}")
    print(f"GRU Velocity shape: {velocity_gru.shape}")
