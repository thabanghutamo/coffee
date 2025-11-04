import numpy as np
import librosa
import pretty_midi
import os
from pathlib import Path


class AudioMIDIDatasetPreprocessor:
    """
    Preprocesses audio and MIDI files for training
    """
    
    def __init__(self, sample_rate=22050, n_mels=128, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return y, sr
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel-spectrogram from audio"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db.T  # Transpose to (time, mel_bins)
    
    def load_midi(self, midi_path):
        """Load MIDI file and extract note information"""
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity,
                    'instrument': instrument.program
                })
        
        return notes
    
    def midi_to_piano_roll(self, midi_path, duration=None):
        """Convert MIDI to piano roll representation"""
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        if duration is None:
            duration = midi_data.get_end_time()
        
        # Create piano roll
        fs = self.sample_rate / self.hop_length
        piano_roll = midi_data.get_piano_roll(fs=fs)
        
        return piano_roll.T  # (time, 128)
    
    def align_audio_midi(self, audio_path, midi_path):
        """Align audio and MIDI for training"""
        audio, sr = self.load_audio(audio_path)
        mel_spec = self.extract_mel_spectrogram(audio)
        
        midi_notes = self.load_midi(midi_path)
        piano_roll = self.midi_to_piano_roll(midi_path)
        
        # Ensure same length
        min_len = min(mel_spec.shape[0], piano_roll.shape[0])
        mel_spec = mel_spec[:min_len]
        piano_roll = piano_roll[:min_len]
        
        return mel_spec, piano_roll, midi_notes
    
    def process_dataset(self, dataset_path, output_path):
        """Process entire dataset"""
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(dataset_path.glob('**/*.wav')) + list(dataset_path.glob('**/*.mp3'))
        
        for i, audio_file in enumerate(audio_files):
            # Find corresponding MIDI file
            midi_file = audio_file.with_suffix('.mid')
            
            if not midi_file.exists():
                continue
            
            try:
                mel_spec, piano_roll, notes = self.align_audio_midi(
                    str(audio_file), str(midi_file))
                
                # Save preprocessed data
                np.savez(
                    output_path / f'sample_{i:05d}.npz',
                    mel_spec=mel_spec,
                    piano_roll=piano_roll,
                    notes=notes
                )
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} files...")
            
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
        
        print(f"Dataset processing complete. Saved to {output_path}")


def extract_timbre_features(audio, sr=22050):
    """Extract timbre features for instrument classification"""
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    
    # Concatenate features
    features = np.vstack([
        mfccs,
        spectral_centroid,
        spectral_rolloff,
        spectral_contrast,
        zcr
    ])
    
    # Average over time
    feature_vector = np.mean(features, axis=1)
    
    return feature_vector


def extract_rhythm_features(audio, sr=22050):
    """Extract rhythm and tempo features"""
    
    # Onset detection
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    
    # Tempo estimation
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    
    # Rhythm patterns
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    
    return {
        'tempo': tempo,
        'beats': beats,
        'onset_envelope': onset_env,
        'tempogram': tempogram
    }


if __name__ == "__main__":
    # Example usage
    preprocessor = AudioMIDIDatasetPreprocessor()
    
    # Process dataset
    # preprocessor.process_dataset('datasets/raw', 'datasets/processed')
    
    print("Preprocessing utilities ready.")
    print("To process your dataset, uncomment the line above and provide paths.")
