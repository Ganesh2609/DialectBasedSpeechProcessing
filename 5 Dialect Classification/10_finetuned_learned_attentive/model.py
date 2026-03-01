"""
Classifier with LEARNED Attentive Pooling.
Model: Tamil Wav2Vec (768 dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedAttentivePooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_len, hidden_size = hidden_states.shape
        scores = self.attention(hidden_states)
        mask = torch.arange(max_len, device=hidden_states.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1)
        pooled = (hidden_states * weights).sum(dim=1)
        return pooled


class DialectClassifierLearnedAttentive(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,  # Tamil Wav2Vec
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
        pooled = self.pooling(embeddings, lengths)
        return self.classifier(pooled)
