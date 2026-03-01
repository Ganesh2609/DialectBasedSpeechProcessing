"""
Dataset for Dialect-Specific ASR Training

Supports filtering by dialect for training dialect-specific models.
Uses random seed 17 for reproducible train/val split.
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
from typing import Dict, List, Tuple, Optional
from transformers import WhisperProcessor
from torch.utils.data import Dataset, DataLoader, random_split

RANDOM_SEED = 17
torch.manual_seed(RANDOM_SEED)


class ASRDataset(Dataset):
    """
    Dataset for ASR training with optional dialect filtering.
    
    Dialect is extracted from the 'name' column: SP{id}_{dialect_code}_{gender}_{num}
    Dialect codes: CH=Chennai, KG=Kongu, S=Srilankan, THA=Thanjavur
    """
    
    # Mapping from dialect code in filename to dialect name
    DIALECT_CODE_MAP = {
        'CH': 'Chennai',
        'KG': 'Kongu',
        'S': 'Srilankan',
        'THA': 'Thanjavur',
    }
    
    def __init__(
        self,
        csv_path: str,
        processor: WhisperProcessor,
        dialect: Optional[str] = None,  # None = all, or "Kongu", "Chennai", "Srilankan", "Thanjavur"
        sampling_rate: int = 16000,
        max_audio_length: int = 640000,
        max_token_length: int = 448,
    ):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.max_token_length = max_token_length
        self.dialect = dialect
        
        # Extract dialect from 'name' column (format: SP{id}_{dialect_code}_{gender}_{num})
        def extract_dialect(name):
            parts = name.split('_')
            if len(parts) >= 2:
                code = parts[1]  # e.g., 'THA', 'CH', 'KG', 'S'
                return self.DIALECT_CODE_MAP.get(code, 'Unknown')
            return 'Unknown'
        
        self.df['dialect'] = self.df['name'].apply(extract_dialect)
        
        # Show dialect distribution
        print(f"Dialect distribution:")
        for d, count in self.df['dialect'].value_counts().items():
            print(f"  {d}: {count}")
        
        # Filter by dialect if specified
        if dialect is not None:
            original_count = len(self.df)
            self.df = self.df[self.df['dialect'] == dialect].reset_index(drop=True)
            print(f"Filtered to dialect '{dialect}': {original_count} -> {len(self.df)} samples")
        else:
            print(f"Using all dialects: {len(self.df)} samples")
        
        # Filter by token length
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
        
        try:
            waveform, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        except Exception as e:
            print(f"\nError loading audio at index {idx}: {audio_path}")
            raise e
        
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        
        return {
            'waveform': waveform,
            'transcription': transcription,
            'audio_id': audio_id,
        }


def preprocess_text(text: str) -> str:
    """Clean transcription text."""
    text = text.replace('\n', ' ').replace('  ', ' ').strip()
    return text


def collate_fn(batch: List[Dict], processor: WhisperProcessor) -> Dict:
    """Collate function for ASR batches."""
    waveforms = [item['waveform'] for item in batch]
    transcriptions = [preprocess_text(item['transcription']) for item in batch]
    audio_ids = [item['audio_id'] for item in batch]
    
    # Process audio - pad to 30 seconds (3000 mel frames)
    audio_inputs = processor.feature_extractor(
        waveforms,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors='pt',
        padding='max_length',
        return_attention_mask=True,
    )
    
    # Tokenize transcriptions
    labels = processor.tokenizer(
        transcriptions,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=448,
    )
    
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
    dialect: Optional[str] = None,
    batch_size: int = 4,
    train_size: float = 0.8,
    num_workers: int = 12,
    prefetch_factor: int = 2,
    max_audio_length: int = 640000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        csv_path: Path to transcripts.csv
        processor: WhisperProcessor
        dialect: Optional dialect filter (None = all)
        batch_size: Batch size
        train_size: Fraction for training
        num_workers: DataLoader workers
        prefetch_factor: Prefetch batches
        max_audio_length: Max audio samples
    
    Returns:
        train_loader, val_loader
    """
    dataset = ASRDataset(
        csv_path=csv_path,
        processor=processor,
        dialect=dialect,
        max_audio_length=max_audio_length,
    )
    
    train_len = int(train_size * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_data, val_data = random_split(dataset, [train_len, val_len], generator=generator)
    
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    collate = partial(collate_fn, processor=processor)
    
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
    dialect: Optional[str] = None,
    batch_size: int = 4,
    train_size: float = 0.8,
    num_workers: int = 4,
    max_audio_length: int = 640000,
) -> DataLoader:
    """Get validation loader only (for evaluation)."""
    dataset = ASRDataset(
        csv_path=csv_path,
        processor=processor,
        dialect=dialect,
        max_audio_length=max_audio_length,
    )
    
    train_len = int(train_size * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, val_data = random_split(dataset, [train_len, val_len], generator=generator)
    
    print(f"Validation samples: {len(val_data)}")
    
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
