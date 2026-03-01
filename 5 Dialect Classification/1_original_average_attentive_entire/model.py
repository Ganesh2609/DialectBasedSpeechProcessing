import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from typing import Optional


class AttentivePooling(nn.Module):
    """
    Learnable attention mechanism for frame-level pooling.
    Computes attention weights over frames and returns weighted sum.
    """
    
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
            hidden_states: Frame embeddings [B, T, D]
            attention_mask: Mask for valid frames [B, T]
        Returns:
            Attentive pooled vector [B, D]
        """
        # Compute attention scores [B, T, 1]
        attention_scores = self.attention(hidden_states)
        
        # Apply mask if provided
        if attention_mask is not None:
            # Convert to [B, T, 1] and set masked positions to -inf
            mask = attention_mask.unsqueeze(-1).float()
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over time dimension
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, T, 1]
        
        # Weighted sum
        pooled = (hidden_states * attention_weights).sum(dim=1)  # [B, D]
        
        return pooled


class Wav2Vec2ForDialectClassification(nn.Module):
    """
    Wav2Vec2/XLSR-53 based model for Tamil dialect classification.
    
    Architecture:
    - Wav2Vec2Model backbone (mostly frozen, top N layers unfrozen)
    - Mean pooling + Attentive pooling → concatenated [B, 2*hidden_size]
    - Classifier MLP: Linear → ReLU → Dropout → Linear → 4 classes
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        num_classes: int = 4,
        num_unfrozen_layers: int = 4,
        classifier_hidden_size: int = 512,
        classifier_dropout: float = 0.3
    ):
        """
        Args:
            model_name: Pretrained Wav2Vec2 model name
            num_classes: Number of dialect classes (default 4)
            num_unfrozen_layers: Number of top encoder layers to unfreeze (default 4)
            classifier_hidden_size: Hidden size of classifier MLP
            classifier_dropout: Dropout probability in classifier
        """
        super().__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = self.wav2vec2.config.hidden_size  # 1024 for large model
        
        # Freeze backbone
        self._freeze_backbone(num_unfrozen_layers)
        
        # Pooling layers
        self.attentive_pooling = AttentivePooling(self.hidden_size)
        
        # Classifier MLP
        # Input: mean_pooled + attentive_pooled = 2 * hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_size, num_classes)
        )
        
        self.num_unfrozen_layers = num_unfrozen_layers
    
    def _freeze_backbone(self, num_unfrozen_layers: int):
        """
        Freeze Wav2Vec2 backbone except top N encoder layers.
        
        Args:
            num_unfrozen_layers: Number of top encoder layers to keep unfrozen
        """
        # Freeze feature extractor (CNN layers)
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
                # Keep unfrozen
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
        Forward pass for dialect classification.
        
        Args:
            input_values: Audio waveform tensor [B, T]
            attention_mask: Attention mask [B, T]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Get Wav2Vec2 outputs
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state  # [B, T', D] where D=1024
        
        # Get reduced attention mask for hidden states
        # Wav2Vec2 reduces temporal dimension, need to compute output mask
        if attention_mask is not None:
            # Compute output attention mask based on input mask
            output_mask = self._get_feat_extract_output_lengths(attention_mask)
        else:
            output_mask = None
        
        # Mean pooling
        if output_mask is not None:
            mask_expanded = output_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean_pooled = sum_hidden / count
        else:
            mean_pooled = hidden_states.mean(dim=1)  # [B, D]
        
        # Attentive pooling
        attentive_pooled = self.attentive_pooling(hidden_states, output_mask)  # [B, D]
        
        # Concatenate
        pooled = torch.cat([mean_pooled, attentive_pooled], dim=-1)  # [B, 2*D]
        
        # Classify
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
    
    def _get_feat_extract_output_lengths(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the output attention mask based on input attention mask.
        The CNN feature extractor reduces the temporal dimension.
        """
        # Get input lengths
        input_lengths = attention_mask.sum(dim=-1)
        
        # Compute output lengths after CNN downsampling
        # Wav2Vec2 CNN has specific kernel sizes and strides
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        
        # Create output mask
        batch_size = attention_mask.size(0)
        max_length = hidden_states_length = output_lengths.max().item()
        
        output_mask = torch.zeros(batch_size, max_length, dtype=torch.long, device=attention_mask.device)
        for i, length in enumerate(output_lengths):
            output_mask[i, :length] = 1
        
        return output_mask
    
    def get_parameter_groups(self, encoder_lr: float = 1e-5, classifier_lr: float = 1e-4):
        """
        Get parameter groups with differential learning rates.
        
        Args:
            encoder_lr: Learning rate for unfrozen encoder layers
            classifier_lr: Learning rate for classifier and pooling layers
            
        Returns:
            List of parameter group dicts for optimizer
        """
        encoder_params = []
        classifier_params = []
        
        # Collect unfrozen encoder parameters
        num_layers = len(self.wav2vec2.encoder.layers)
        freeze_until = num_layers - self.num_unfrozen_layers
        
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i >= freeze_until:
                encoder_params.extend(layer.parameters())
        
        # Add layer norm
        encoder_params.extend(self.wav2vec2.encoder.layer_norm.parameters())
        
        # Classifier and pooling parameters
        classifier_params.extend(self.attentive_pooling.parameters())
        classifier_params.extend(self.classifier.parameters())
        
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': classifier_params, 'lr': classifier_lr}
        ]
