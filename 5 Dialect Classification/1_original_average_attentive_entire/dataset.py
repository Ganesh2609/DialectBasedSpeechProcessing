import os
import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import librosa
import numpy as np
import pandas as pd
from functools import partial
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor
from torch.utils.data import Dataset, DataLoader, random_split
import torch

# Set random seed for reproducibility
RANDOM_SEED = 17
torch.manual_seed(RANDOM_SEED)


# Dialect label mapping
DIALECT_LABELS = {
    'Central_Dialect': 0,
    'Northern_Dialect': 1,
    'Southern_Dialect': 2,
    'Western_Dialect': 3
}

LABEL_TO_DIALECT = {v: k for k, v in DIALECT_LABELS.items()}


class DialectDataset(Dataset):
    """
    Dataset for Tamil Dialect Classification.
    Loads audio files and extracts dialect labels from folder structure.
    """

    def __init__(self, csv_path: str, feature_extractor: Wav2Vec2FeatureExtractor, sampling_rate: int = 16000, max_length: int = 160000):
        """
        Args:
            csv_path: Path to transcripts.csv containing audio_path column
            feature_extractor: Wav2Vec2FeatureExtractor for processing audio
            sampling_rate: Target sampling rate (default 16kHz for Wav2Vec2)
            max_length: Maximum audio length in samples (default 10 seconds at 16kHz)
        """
        self.df = pd.read_csv(csv_path)
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        
        # Extract dialect labels from audio_path
        self.labels = []
        for audio_path in self.df['audio_path']:
            dialect = self._extract_dialect(audio_path)
            self.labels.append(DIALECT_LABELS[dialect])
    
    def _extract_dialect(self, audio_path: str) -> str:
        """Extract dialect name from audio path."""
        path_parts = audio_path.replace('\\', '/').split('/')
        for part in path_parts:
            if part in DIALECT_LABELS:
                return part
        raise ValueError(f"Could not extract dialect from path: {audio_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        audio_path = self.df.iloc[idx]['audio_path']
        label = self.labels[idx]
        
        try:
            # Load audio using librosa (handles resampling and mono conversion)
            waveform, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        except Exception as e:
            print(f"\nError loading file at index {idx}: {audio_path}")
            print(f"Error type: {type(e).__name__}")
            if not os.path.exists(audio_path):
                print(f"File does not exist!")
            raise
        
        # Truncate if too long
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        
        return {
            'waveform': waveform,
            'label': label,
        }


def collate_fn(batch, feature_extractor: Wav2Vec2FeatureExtractor):
    """
    Collate function for dynamic padding of audio sequences.
    """
    waveforms = [item['waveform'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    # Process with feature extractor (handles padding, no normalization since already preprocessed)
    inputs = feature_extractor(
        waveforms,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors='pt',
        padding=True,
        return_attention_mask=True,
        do_normalize=False  # Skip normalization - audio already LUFS normalized
    )
    
    return {
        'input_values': inputs['input_values'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels
    }


def get_data_loaders(
    csv_path: str, 
    feature_extractor: Wav2Vec2FeatureExtractor, 
    batch_size: int = 2, 
    train_size: float = 0.8, 
    num_workers: int = 4, 
    prefetch_factor: int = 2,
    max_length: int = 160000
):
    """
    Create train and validation data loaders for dialect classification.
    
    Args:
        csv_path: Path to transcripts.csv
        feature_extractor: Wav2Vec2FeatureExtractor instance
        batch_size: Batch size for data loaders
        train_size: Fraction of data for training (0.0 to 1.0)
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of batches to prefetch per worker
        max_length: Maximum audio length in samples
        
    Returns:
        train_loader, val_loader: DataLoader instances
    """
    dataset = DialectDataset(
        csv_path=csv_path, 
        feature_extractor=feature_extractor,
        max_length=max_length
    )
    
    # Split dataset
    train_len = int(train_size * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_data, val_data = random_split(dataset, [train_len, val_len], generator=generator)
    
    # Create data loaders
    collate = partial(collate_fn, feature_extractor=feature_extractor)

    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    # val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate
    )
    
    return train_loader, val_loader
