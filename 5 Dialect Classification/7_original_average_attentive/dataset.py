"""
Dataset for loading pre-extracted features from .npz file.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# Set random seed for reproducibility
RANDOM_SEED = 17
torch.manual_seed(RANDOM_SEED)


class FeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted Wav2Vec2 features.
    """
    
    def __init__(self, npz_path: str):
        """
        Args:
            npz_path: Path to .npz file containing features
        """
        data = np.load(npz_path, allow_pickle=True)
        self.names = data['names']
        self.audio_paths = data['audio_paths']
        self.embeddings = data['embeddings'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        
        print(f"Loaded {len(self.labels)} samples with embedding dim {self.embeddings.shape[1]}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'embedding': torch.from_numpy(self.embeddings[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def collate_fn(batch):
    """Simple collate function for feature batches."""
    embeddings = torch.stack([item['embedding'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {
        'embeddings': embeddings,
        'labels': labels
    }


def get_data_loaders(
    npz_path: str,
    batch_size: int = 32,
    train_size: float = 0.8,
    num_workers: int = 0
):
    """
    Create train and validation data loaders.
    """
    dataset = FeatureDataset(npz_path)
    
    # Split dataset with fixed seed
    train_len = int(train_size * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_data, val_data = random_split(dataset, [train_len, val_len], generator=generator)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader
