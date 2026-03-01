"""
Dataset for Standard ASR Training (No Dialect Conditioning)

Loads:
- Audio files from Final Dataset/Train/
- Transcriptions from transcripts.csv

Uses random seed 17 for train/val split (matching dialect classification).
"""

import os
import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import librosa
import numpy as np
import pandas as pd
from functools import partial
from typing import Dict, List, Tuple
from transformers import WhisperProcessor
from torch.utils.data import Dataset, DataLoader, random_split

# Set random seed for reproducibility
RANDOM_SEED = 17
torch.manual_seed(RANDOM_SEED)


class ASRDataset(Dataset):
    """
    Dataset for standard ASR training without dialect conditioning.
    
    Loads audio and transcriptions only.
    Filters out samples exceeding max token length (Whisper's 448 token limit).
    """
    
    def __init__(
        self,
        csv_path: str,
        processor: WhisperProcessor,
        sampling_rate: int = 16000,
        max_audio_length: int = 640000,  # 40 seconds at 16kHz
        max_token_length: int = 448,  # Whisper's decoder limit
    ):
        """
        Args:
            csv_path: Path to transcripts.csv
            processor: WhisperProcessor for audio and text processing
            sampling_rate: Target sampling rate (16kHz for Whisper)
            max_audio_length: Maximum audio length in samples
            max_token_length: Maximum token length for transcriptions (Whisper limit: 448)
        """
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.max_token_length = max_token_length
        
        print(f"Loaded {len(self.df)} samples from CSV")
        
        # Filter out samples exceeding max token length
        valid_indices = []
        excluded_count = 0
        for idx in range(len(self.df)):
            transcription = str(self.df.iloc[idx]['transcription'])
            tokens = processor.tokenizer(transcription, return_tensors='pt')
            token_len = tokens['input_ids'].shape[1]
            if token_len <= max_token_length:
                valid_indices.append(idx)
            else:
                excluded_count += 1
        
        if excluded_count > 0:
            print(f"Excluded {excluded_count} samples exceeding {max_token_length} token limit")
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
            print(f"Remaining samples: {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict:
        if torch.is_tensor(idx):
            idx = idx.item()
        
        row = self.df.iloc[idx]
        audio_path = row['audio_path']
        audio_id = row['name']
        transcription = str(row['transcription'])
        
        # Load audio
        try:
            waveform, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        except Exception as e:
            print(f"\nError loading audio at index {idx}: {audio_path}")
            raise e
        
        # Truncate if too long
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        
        return {
            'waveform': waveform,
            'transcription': transcription,
            'audio_id': audio_id,
        }


def preprocess_text(text: str) -> str:
    """Clean transcription text."""
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')
    text = text.strip()
    return text


def collate_fn(batch: List[Dict], processor: WhisperProcessor) -> Dict:
    """
    Collate function for standard ASR.
    
    Returns:
        - input_features: Whisper mel spectrogram [B, n_mels, seq_len]
        - labels: Tokenized transcriptions [B, seq_len]
    """
    waveforms = [item['waveform'] for item in batch]
    transcriptions = [preprocess_text(item['transcription']) for item in batch]
    audio_ids = [item['audio_id'] for item in batch]
    
    # Process audio to mel spectrogram - pad to 30 seconds (3000 mel frames)
    audio_inputs = processor.feature_extractor(
        waveforms,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors='pt',
        padding='max_length',  # Pad to max_length (30 seconds = 3000 mel frames)
        return_attention_mask=True,
    )
    
    # Tokenize transcriptions
    labels = processor.tokenizer(
        transcriptions,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=448,  # Whisper max token length
    )
    
    # Convert labels: replace padding token id with -100 for loss computation
    label_ids = labels['input_ids']
    label_ids[label_ids == processor.tokenizer.pad_token_id] = -100
    
    return {
        'input_features': audio_inputs['input_features'],
        'labels': label_ids,
        'transcription_text': transcriptions,
        'audio_ids': audio_ids,
    }


def get_data_loaders(
    csv_path: str,
    processor: WhisperProcessor,
    batch_size: int = 4,
    train_size: float = 0.8,
    num_workers: int = 12,
    prefetch_factor: int = 2,
    max_audio_length: int = 640000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders for standard ASR.
    
    Uses random seed 17 for reproducible split matching dialect classification.
    
    Args:
        csv_path: Path to transcripts.csv
        processor: WhisperProcessor instance
        batch_size: Batch size for data loaders
        train_size: Fraction of data for training
        num_workers: Number of worker processes
        prefetch_factor: Batches to prefetch per worker
        max_audio_length: Maximum audio length in samples
        
    Returns:
        train_loader, val_loader
    """
    # Create dataset
    dataset = ASRDataset(
        csv_path=csv_path,
        processor=processor,
        max_audio_length=max_audio_length,
    )
    
    # Split dataset with fixed seed
    train_len = int(train_size * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_data, val_data = random_split(dataset, [train_len, val_len], generator=generator)
    
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Create collate function
    collate = partial(collate_fn, processor=processor)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate,
    )
    
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate,
    )
    
    return train_loader, val_loader


def get_validation_loader(
    csv_path: str,
    processor: WhisperProcessor,
    batch_size: int = 4,
    train_size: float = 0.8,
    num_workers: int = 4,
    max_audio_length: int = 640000,
) -> DataLoader:
    """
    Get only the validation loader (for beam search evaluation).
    
    Uses same random seed 17 to ensure identical split.
    """
    # Create dataset
    dataset = ASRDataset(
        csv_path=csv_path,
        processor=processor,
        max_audio_length=max_audio_length,
    )
    
    # Split with same seed
    train_len = int(train_size * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, val_data = random_split(dataset, [train_len, val_len], generator=generator)
    
    print(f"Validation samples: {len(val_data)}")
    
    # Create collate function
    collate = partial(collate_fn, processor=processor)
    
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate,
    )
    
    return val_loader
