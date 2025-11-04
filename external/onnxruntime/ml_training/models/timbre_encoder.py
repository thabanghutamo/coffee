import torch
import torch.nn as nn
import torch.nn.functional as F

class TimbreAutoencoder(nn.Module):
    """
    Autoencoder for learning timbre representations from audio loops
    Uses contrastive learning to map similar instruments to nearby embeddings
    """
    
    def __init__(self, input_dim=128, latent_dim=64):
        super(TimbreAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning timbre similarities
    Pulls similar instruments together, pushes different ones apart
    """
    
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch, latent_dim)
            labels: (batch,) instrument class labels
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive pairs mask
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Average over positive pairs
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-6)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


class InstrumentClassifier(nn.Module):
    """
    Classifier to predict instrument type from timbre embedding
    """
    
    def __init__(self, latent_dim=64, num_instruments=20):
        super(InstrumentClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_instruments)
        )
        
    def forward(self, z):
        return self.classifier(z)


class TimbreMapper:
    """
    Maps user voice/beatbox input to instrument timbres using trained autoencoder
    """
    
    def __init__(self, autoencoder_path=None):
        self.autoencoder = TimbreAutoencoder()
        
        if autoencoder_path:
            self.load_model(autoencoder_path)
        
        # Precomputed instrument embeddings (to be loaded from training)
        self.instrument_embeddings = {}
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.autoencoder.eval()
        
        if 'instrument_embeddings' in checkpoint:
            self.instrument_embeddings = checkpoint['instrument_embeddings']
    
    def extract_timbre_features(self, audio_features):
        """
        Extract timbre embedding from audio features
        Args:
            audio_features: MFCC or mel-spectrogram features
        Returns:
            timbre_embedding: (latent_dim,)
        """
        with torch.no_grad():
            features_tensor = torch.tensor(audio_features, dtype=torch.float32)
            embedding = self.autoencoder.encode(features_tensor)
        
        return embedding.numpy()
    
    def find_closest_instrument(self, voice_embedding, top_k=3):
        """
        Find closest matching instruments for voice input
        """
        similarities = {}
        voice_emb_norm = voice_embedding / (torch.norm(voice_embedding) + 1e-8)
        
        for instrument_name, inst_embedding in self.instrument_embeddings.items():
            inst_emb_norm = inst_embedding / (torch.norm(inst_embedding) + 1e-8)
            similarity = torch.dot(voice_emb_norm, inst_emb_norm).item()
            similarities[instrument_name] = similarity
        
        # Sort by similarity
        sorted_instruments = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_instruments[:top_k]


if __name__ == "__main__":
    # Test timbre autoencoder
    print("Testing Timbre Autoencoder...")
    model = TimbreAutoencoder(input_dim=128, latent_dim=64)
    test_input = torch.randn(16, 128)
    reconstruction, embedding = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Test contrastive loss
    print("\nTesting Contrastive Loss...")
    labels = torch.randint(0, 5, (16,))
    contrastive_loss_fn = ContrastiveLoss()
    loss = contrastive_loss_fn(embedding, labels)
    print(f"Contrastive loss: {loss.item():.4f}")
    
    # Test instrument classifier
    print("\nTesting Instrument Classifier...")
    classifier = InstrumentClassifier(latent_dim=64, num_instruments=20)
    predictions = classifier(embedding)
    print(f"Predictions shape: {predictions.shape}")
