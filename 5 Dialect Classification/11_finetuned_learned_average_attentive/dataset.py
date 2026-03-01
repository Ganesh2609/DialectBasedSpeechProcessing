"""
Dataset for loading frame-level features with variable lengths.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Set random seed for reproducibility
RANDOM_SEED = 17
torch.manual_seed(RANDOM_SEED)


class FrameFeatureDataset(Dataset):
    """
    Dataset for frame-level Wav2Vec2 features with lengths for masking.
    """
    
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.names = data['names']
        self.audio_paths = data['audio_paths']
        self.embeddings = data['embeddings'].astype(np.float32)  # [N, T, D]
        self.lengths = data['lengths'].astype(np.int32)
        self.labels = data['labels'].astype(np.int64)
        
        print(f"Loaded {len(self.labels)} samples, max_len={self.embeddings.shape[1]}, dim={self.embeddings.shape[2]}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        length = self.lengths[idx]
        # Only return actual frames (not padding)
        embedding = self.embeddings[idx, :length, :]  # [T, D]
        return {
            'embedding': torch.from_numpy(embedding),
            'length': length,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def collate_fn(batch):
    """Collate with dynamic padding and length tracking."""
    embeddings = [item['embedding'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    labels = torch.stack([item['label'] for item in batch])
    
    # Pad sequences to max length in batch
    padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    
    return {
        'embeddings': padded,  # [B, T, D]
        'lengths': lengths,     # [B]
        'labels': labels        # [B]
    }


def get_data_loaders(
    npz_path: str,
    batch_size: int = 32,
    train_size: float = 0.8,
    num_workers: int = 0
):
    dataset = FrameFeatureDataset(npz_path)
    
    train_len = int(train_size * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_data, val_data = random_split(dataset, [train_len, val_len], generator=generator)
    
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        dataset=val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    return train_loader, val_loader
