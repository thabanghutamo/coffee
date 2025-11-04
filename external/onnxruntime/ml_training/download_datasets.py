#!/usr/bin/env python3
"""
Dataset Collection Script for Vocal MIDI Generator
Downloads and preprocesses Lakh MIDI, MAESTRO, and NSynth datasets
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import json

# Dataset URLs
DATASETS = {
    'lakh': {
        'name': 'Lakh MIDI Dataset',
        'url': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz',
        'size': '~2.5GB',
        'description': '176,581 unique MIDI files from various genres'
    },
    'maestro': {
        'name': 'MAESTRO Dataset v3.0.0',
        'url': 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
        'size': '~65MB',
        'description': '1,282 piano performances from competition recordings'
    },
    'nsynth': {
        'name': 'NSynth Dataset',
        'url': 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz',
        'size': '~30GB',
        'description': '305,979 audio samples from 1,006 instruments'
    }
}

class DatasetDownloader:
    def __init__(self, output_dir='./datasets'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url, destination):
        """Download file with progress bar"""
        print(f"Downloading from {url}")
        
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=destination.name) as t:
            urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
    
    def extract_archive(self, archive_path, extract_to):
        """Extract tar.gz or zip archive"""
        print(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.name.endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False
        
        return True
    
    def download_lakh(self):
        """Download and extract Lakh MIDI Dataset"""
        dataset_dir = self.output_dir / 'lakh'
        dataset_dir.mkdir(exist_ok=True)
        
        archive_path = dataset_dir / 'lmd_full.tar.gz'
        
        if not archive_path.exists():
            self.download_file(DATASETS['lakh']['url'], archive_path)
        
        # Extract
        if not (dataset_dir / 'lmd_full').exists():
            self.extract_archive(archive_path, dataset_dir)
        
        print(f"✓ Lakh MIDI Dataset ready at {dataset_dir}")
        return dataset_dir / 'lmd_full'
    
    def download_maestro(self):
        """Download and extract MAESTRO Dataset"""
        dataset_dir = self.output_dir / 'maestro'
        dataset_dir.mkdir(exist_ok=True)
        
        archive_path = dataset_dir / 'maestro-v3.0.0-midi.zip'
        
        if not archive_path.exists():
            self.download_file(DATASETS['maestro']['url'], archive_path)
        
        # Extract
        if not (dataset_dir / 'maestro-v3.0.0').exists():
            self.extract_archive(archive_path, dataset_dir)
        
        print(f"✓ MAESTRO Dataset ready at {dataset_dir}")
        return dataset_dir / 'maestro-v3.0.0'
    
    def download_nsynth(self, subset='train'):
        """Download and extract NSynth Dataset (train/valid/test)"""
        dataset_dir = self.output_dir / 'nsynth'
        dataset_dir.mkdir(exist_ok=True)
        
        # NSynth is very large, only download if explicitly requested
        print(f"Warning: NSynth dataset is ~30GB. This will take a while.")
        response = input("Continue? (y/n): ")
        
        if response.lower() != 'y':
            print("Skipping NSynth download")
            return None
        
        url = f"http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{subset}.jsonwav.tar.gz"
        archive_path = dataset_dir / f'nsynth-{subset}.tar.gz'
        
        if not archive_path.exists():
            self.download_file(url, archive_path)
        
        # Extract
        if not (dataset_dir / f'nsynth-{subset}').exists():
            self.extract_archive(archive_path, dataset_dir)
        
        print(f"✓ NSynth Dataset ({subset}) ready at {dataset_dir}")
        return dataset_dir / f'nsynth-{subset}'
    
    def verify_datasets(self):
        """Verify downloaded datasets"""
        results = {}
        
        # Check Lakh
        lakh_path = self.output_dir / 'lakh' / 'lmd_full'
        if lakh_path.exists():
            midi_files = list(lakh_path.rglob('*.mid')) + list(lakh_path.rglob('*.midi'))
            results['lakh'] = {'exists': True, 'files': len(midi_files)}
        else:
            results['lakh'] = {'exists': False}
        
        # Check MAESTRO
        maestro_path = self.output_dir / 'maestro' / 'maestro-v3.0.0'
        if maestro_path.exists():
            midi_files = list(maestro_path.rglob('*.midi')) + list(maestro_path.rglob('*.mid'))
            results['maestro'] = {'exists': True, 'files': len(midi_files)}
        else:
            results['maestro'] = {'exists': False}
        
        # Check NSynth
        nsynth_path = self.output_dir / 'nsynth' / 'nsynth-train'
        if nsynth_path.exists():
            audio_files = list(nsynth_path.rglob('*.wav'))
            results['nsynth'] = {'exists': True, 'files': len(audio_files)}
        else:
            results['nsynth'] = {'exists': False}
        
        return results

def create_dataset_manifest(datasets_dir, output_file='dataset_manifest.json'):
    """Create manifest file listing all available data"""
    manifest = {
        'lakh_midi': [],
        'maestro_midi': [],
        'nsynth_audio': []
    }
    
    datasets_path = Path(datasets_dir)
    
    # Scan Lakh MIDI
    lakh_path = datasets_path / 'lakh' / 'lmd_full'
    if lakh_path.exists():
        for midi_file in lakh_path.rglob('*.mid'):
            manifest['lakh_midi'].append(str(midi_file.relative_to(datasets_path)))
        for midi_file in lakh_path.rglob('*.midi'):
            manifest['lakh_midi'].append(str(midi_file.relative_to(datasets_path)))
    
    # Scan MAESTRO
    maestro_path = datasets_path / 'maestro' / 'maestro-v3.0.0'
    if maestro_path.exists():
        for midi_file in maestro_path.rglob('*.midi'):
            manifest['maestro_midi'].append(str(midi_file.relative_to(datasets_path)))
        for midi_file in maestro_path.rglob('*.mid'):
            manifest['maestro_midi'].append(str(midi_file.relative_to(datasets_path)))
    
    # Scan NSynth
    nsynth_path = datasets_path / 'nsynth' / 'nsynth-train'
    if nsynth_path.exists():
        for audio_file in nsynth_path.rglob('*.wav'):
            manifest['nsynth_audio'].append(str(audio_file.relative_to(datasets_path)))
    
    # Save manifest
    manifest_path = datasets_path / output_file
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Created dataset manifest at {manifest_path}")
    print(f"  Lakh MIDI files: {len(manifest['lakh_midi'])}")
    print(f"  MAESTRO MIDI files: {len(manifest['maestro_midi'])}")
    print(f"  NSynth audio files: {len(manifest['nsynth_audio'])}")
    
    return manifest_path

def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare datasets for Vocal MIDI Generator training'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./datasets',
        help='Output directory for datasets (default: ./datasets)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        choices=['lakh', 'maestro', 'nsynth', 'all'],
        default=['all'],
        help='Which datasets to download (default: all)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing datasets without downloading'
    )
    parser.add_argument(
        '--create-manifest',
        action='store_true',
        help='Create dataset manifest file'
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output_dir)
    
    # Verify only mode
    if args.verify_only:
        print("Verifying datasets...")
        results = downloader.verify_datasets()
        
        for dataset, info in results.items():
            if info['exists']:
                print(f"✓ {dataset}: {info['files']} files")
            else:
                print(f"✗ {dataset}: not found")
        
        return
    
    # Download datasets
    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['lakh', 'maestro']  # Skip nsynth by default due to size
    
    print("=" * 60)
    print("Vocal MIDI Generator - Dataset Downloader")
    print("=" * 60)
    print()
    
    for dataset in datasets_to_download:
        print(f"\n--- {DATASETS[dataset]['name']} ---")
        print(f"Size: {DATASETS[dataset]['size']}")
        print(f"Description: {DATASETS[dataset]['description']}")
        print()
        
        if dataset == 'lakh':
            downloader.download_lakh()
        elif dataset == 'maestro':
            downloader.download_maestro()
        elif dataset == 'nsynth':
            downloader.download_nsynth()
    
    # Verify downloads
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    results = downloader.verify_datasets()
    
    for dataset, info in results.items():
        if info['exists']:
            print(f"✓ {dataset}: {info['files']} files")
        else:
            print(f"✗ {dataset}: not found")
    
    # Create manifest if requested
    if args.create_manifest:
        create_dataset_manifest(args.output_dir)
    
    print("\n✓ Dataset preparation complete!")
    print(f"Datasets are located in: {Path(args.output_dir).absolute()}")
    print("\nNext steps:")
    print("1. Run preprocessing: python ml_training/utils/data_preprocessing.py")
    print("2. Start training: python ml_training/train.py")

if __name__ == '__main__':
    main()
