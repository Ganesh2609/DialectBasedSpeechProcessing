"""
Simple Classifier Model for pre-extracted Wav2Vec2 features.
Pooling: average_attentive (both poolings concatenated) - input dim 2048
"""

import torch
import torch.nn as nn


class DialectClassifier(nn.Module):
    """
    MLP Classifier for dialect classification on pre-extracted features.
    """
    
    def __init__(
        self,
        input_dim: int = 1536,  # 1536 for Tamil Wav2Vec average_attentive (768 + 768)
        hidden_dim: int = 512,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Pre-extracted features [B, input_dim]
        Returns:
            logits: [B, num_classes]
        """
        return self.classifier(embeddings)
