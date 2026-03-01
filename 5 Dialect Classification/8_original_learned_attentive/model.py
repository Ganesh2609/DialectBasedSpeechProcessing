"""
Classifier with LEARNED Attentive Pooling.
Model: facebook/wav2vec2-large-xlsr-53 (1024 dim)
Pooling: learned_attentive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedAttentivePooling(nn.Module):
    """
    Learned attention pooling with trainable attention weights.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D]
            lengths: [B] actual lengths for masking
        Returns:
            pooled: [B, D]
        """
        batch_size, max_len, hidden_size = hidden_states.shape
        
        # Compute attention scores
        scores = self.attention(hidden_states)  # [B, T, 1]
        
        # Create mask for padding
        mask = torch.arange(max_len, device=hidden_states.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # [B, T, 1]
        
        # Mask out padding positions
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax over valid positions
        weights = F.softmax(scores, dim=1)  # [B, T, 1]
        
        # Weighted sum
        pooled = (hidden_states * weights).sum(dim=1)  # [B, D]
        
        return pooled


class DialectClassifierLearnedAttentive(nn.Module):
    """
    Classifier with learned attentive pooling.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # 1024 for Facebook XLSR
        hidden_dim: int = 512,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.pooling = LearnedAttentivePooling(input_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, T, D] frame-level features
            lengths: [B] actual lengths
        Returns:
            logits: [B, num_classes]
        """
        pooled = self.pooling(embeddings, lengths)  # [B, D]
        return self.classifier(pooled)
