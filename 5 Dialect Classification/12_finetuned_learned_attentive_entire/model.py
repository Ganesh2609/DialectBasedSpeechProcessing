"""
Tamil Wav2Vec2 with Learned Attentive Pooling and Transformer Fine-tuning
Model: Harveenchadha/vakyansh-wav2vec2-tamil-tam-250 (768 dim)
Pooling: Learned Attentive only (not average+attentive)
Fine-tunes top N transformer layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from typing import Optional


class LearnedAttentivePooling(nn.Module):
    """Learnable attention mechanism for frame-level pooling."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1, bias=False)
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            hidden_states: [B, T, D]
            attention_mask: [B, T] mask for valid frames
        Returns:
            pooled: [B, D]
        """
        attention_scores = self.attention(hidden_states)  # [B, T, 1]
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=1)
        pooled = (hidden_states * attention_weights).sum(dim=1)
        
        return pooled


class TamilWav2VecLearnedAttentive(nn.Module):
    """
    Tamil Wav2Vec2 with learned attentive pooling.
    Fine-tunes top N transformer layers.
    """
    
    def __init__(
        self, 
        model_name: str = "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
        num_classes: int = 4,
        num_unfrozen_layers: int = 4,
        classifier_hidden_size: int = 512,
        classifier_dropout: float = 0.3
    ):
        super().__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = self.wav2vec2.config.hidden_size  # 768 for Tamil model
        
        # Freeze backbone except top N layers
        self._freeze_backbone(num_unfrozen_layers)
        
        # Learned attentive pooling only (not average+attentive)
        self.attentive_pooling = LearnedAttentivePooling(self.hidden_size)
        
        # Classifier MLP - input is hidden_size (768), not 2*hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_size, classifier_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_size // 2, num_classes)
        )
        
        self.num_unfrozen_layers = num_unfrozen_layers
    
    def _freeze_backbone(self, num_unfrozen_layers: int):
        """Freeze Wav2Vec2 backbone except top N encoder layers."""
        # Freeze feature extractor (CNN)
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        
        # Freeze feature projection
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False
        
        # Freeze positional conv embedding
        for param in self.wav2vec2.encoder.pos_conv_embed.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers except top N
        num_layers = len(self.wav2vec2.encoder.layers)
        freeze_until = num_layers - num_unfrozen_layers
        
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Keep layer norm unfrozen
        for param in self.wav2vec2.encoder.layer_norm.parameters():
            param.requires_grad = True
    
    def forward(
        self, 
        input_values: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_values: Audio waveform [B, T]
            attention_mask: [B, T]
        Returns:
            logits: [B, num_classes]
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state  # [B, T', D]
        
        # Get reduced attention mask
        if attention_mask is not None:
            output_mask = self._get_feat_extract_output_lengths(attention_mask)
        else:
            output_mask = None
        
        # Only attentive pooling (no mean pooling)
        pooled = self.attentive_pooling(hidden_states, output_mask)  # [B, D]
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def _get_feat_extract_output_lengths(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute output attention mask after CNN downsampling."""
        input_lengths = attention_mask.sum(dim=-1)
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        
        batch_size = attention_mask.size(0)
        max_length = output_lengths.max().item()
        
        output_mask = torch.zeros(batch_size, max_length, dtype=torch.long, device=attention_mask.device)
        for i, length in enumerate(output_lengths):
            output_mask[i, :length] = 1
        
        return output_mask
    
    def get_parameter_groups(self, encoder_lr: float = 1e-5, classifier_lr: float = 1e-4):
        """Get parameter groups with differential learning rates."""
        encoder_params = []
        classifier_params = []
        
        num_layers = len(self.wav2vec2.encoder.layers)
        freeze_until = num_layers - self.num_unfrozen_layers
        
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i >= freeze_until:
                encoder_params.extend(layer.parameters())
        
        encoder_params.extend(self.wav2vec2.encoder.layer_norm.parameters())
        
        classifier_params.extend(self.attentive_pooling.parameters())
        classifier_params.extend(self.classifier.parameters())
        
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': classifier_params, 'lr': classifier_lr}
        ]
