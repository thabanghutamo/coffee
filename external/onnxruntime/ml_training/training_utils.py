import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from pathlib import Path
import json
import time
from datetime import datetime


class TrainingConfig:
    """Centralized training configuration"""
    
    def __init__(self, config_dict=None):
        # Model architecture
        self.mel_bins = 128
        self.hidden_size = 256
        self.num_layers = 2
        
        # Training
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.grad_clip = 1.0
        
        # Distributed training
        self.distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        
        # Checkpointing
        self.checkpoint_dir = './checkpoints'
        self.save_every = 5  # Save every N epochs
        self.keep_last_n = 5  # Keep only last N checkpoints
        
        # Logging
        self.use_wandb = True
        self.wandb_project = 'vocal-midi-generator'
        self.log_every = 10  # Log every N batches
        
        # Data
        self.data_dir = './datasets'
        self.num_workers = 4
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Override with config dict
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup"""
    
    def __init__(self, checkpoint_dir, keep_last_n=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, model_name='model'):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'{model_name}_epoch_{epoch:04d}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(model_name)
        
        return checkpoint_path
    
    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None, model_name='model'):
        """Load model checkpoint"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob(f'{model_name}_epoch_*.pt'))
            if not checkpoints:
                print(f"No checkpoints found for {model_name}")
                return 0
            checkpoint_path = checkpoints[-1]
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        print(f"✓ Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def _cleanup_old_checkpoints(self, model_name):
        """Remove old checkpoints, keeping only last N"""
        checkpoints = sorted(self.checkpoint_dir.glob(f'{model_name}_epoch_*.pt'))
        
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                checkpoint.unlink()
                print(f"Removed old checkpoint: {checkpoint}")


class MetricsLogger:
    """Handles metrics logging to console, file, and W&B"""
    
    def __init__(self, config, experiment_name=None):
        self.config = config
        self.experiment_name = experiment_name or f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Create logs directory
        self.log_dir = Path('./logs') / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        if config.use_wandb and config.rank == 0:
            wandb.init(
                project=config.wandb_project,
                name=self.experiment_name,
                config=vars(config)
            )
        
        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def log(self, metrics_dict, step=None, commit=True):
        """Log metrics"""
        # Store metrics
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Log to W&B
        if self.config.use_wandb and self.config.rank == 0:
            wandb.log(metrics_dict, step=step, commit=commit)
        
        # Console output
        if self.config.rank == 0:
            metrics_str = ' | '.join([f'{k}: {v:.6f}' for k, v in metrics_dict.items()])
            print(f"[Step {step}] {metrics_str}")
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = self.log_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Saved metrics to {metrics_file}")
    
    def finish(self):
        """Cleanup and finalize logging"""
        self.save_metrics()
        if self.config.use_wandb and self.config.rank == 0:
            wandb.finish()


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop


def get_data_loaders(config):
    """Create data loaders with optional distributed sampling"""
    from train import MIDIDataset  # Import from original train.py
    
    train_dataset = MIDIDataset(config.data_dir, split='train')
    val_dataset = MIDIDataset(config.data_dir, split='val')
    
    if config.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=config.world_size,
            rank=config.rank
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model, config):
    """Create optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    return optimizer, scheduler


def save_onnx_model(model, output_path, input_shape):
    """Export trained model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    print(f"✓ Exported ONNX model: {output_path}")
