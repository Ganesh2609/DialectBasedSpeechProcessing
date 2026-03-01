"""
Dialect-Conditioned Whisper ASR Model (Small)

Injects pre-extracted dialect embeddings into Whisper encoder's residual stream.

Model: vasista22/whisper-tamil-small
Dialect Embedding: 768-dim from TamilWav2Vec learned attentive pooling
Whisper Hidden Size: 768 (for small model)

Strategy: Project 768 → 768, add to encoder hidden states at each layer.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput


class DialectConditionedWhisperEncoder(nn.Module):
    """Whisper encoder with dialect embedding injection at each layer."""
    
    def __init__(self, whisper_encoder, dialect_projection: nn.Module):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.dialect_projection = dialect_projection
    
    def forward(
        self,
        input_features: torch.Tensor,
        dialect_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_features: Mel spectrogram [B, n_mels, seq_len]
            dialect_embedding: Pre-extracted dialect embedding [B, 768]
            attention_mask: Optional attention mask [B, seq_len]
        Returns:
            BaseModelOutput with last_hidden_state
        """
        # Conv layers
        inputs_embeds = self.whisper_encoder.conv1(input_features)
        inputs_embeds = nn.functional.gelu(inputs_embeds)
        inputs_embeds = self.whisper_encoder.conv2(inputs_embeds)
        inputs_embeds = nn.functional.gelu(inputs_embeds)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # [B, seq_len, d_model]
        
        # Positional embeddings
        embed_pos = self.whisper_encoder.embed_positions.weight
        hidden_states = inputs_embeds + embed_pos[:inputs_embeds.size(1)]
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.whisper_encoder.dropout, training=self.training
        )
        
        # Project dialect embedding: [B, 768] → [B, 1, 1024]
        projected_dialect = self.dialect_projection(dialect_embedding).unsqueeze(1)
        
        # Pass through encoder layers with dialect injection
        for encoder_layer in self.whisper_encoder.layers:
            # Inject dialect embedding into residual stream
            hidden_states = hidden_states + projected_dialect
            
            layer_outputs = encoder_layer(hidden_states, attention_mask=attention_mask, layer_head_mask=None)
            hidden_states = layer_outputs[0]
        
        # Final layer norm
        hidden_states = self.whisper_encoder.layer_norm(hidden_states)
        
        return BaseModelOutput(last_hidden_state=hidden_states)


class DialectConditionedWhisper(nn.Module):
    """
    Whisper ASR conditioned on dialect embeddings.
    Injects dialect embeddings into encoder residual stream at each layer.
    """
    
    def __init__(
        self,
        whisper_model_name: str = "vasista22/whisper-tamil-medium",
        dialect_embedding_dim: int = 768,
    ):
        super().__init__()
        
        # Load pretrained Whisper
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        self.config = self.whisper.config
        self.encoder_hidden_size = self.config.d_model  # 1024 for medium
        
        # Dialect projection: 768 → 1024
        self.dialect_projection = nn.Sequential(
            nn.Linear(dialect_embedding_dim, self.encoder_hidden_size),
            nn.LayerNorm(self.encoder_hidden_size),
        )
        
        # Wrap encoder with dialect conditioning
        self.dialect_encoder = DialectConditionedWhisperEncoder(
            whisper_encoder=self.whisper.model.encoder,
            dialect_projection=self.dialect_projection
        )
    
    def forward(
        self,
        input_features: torch.Tensor,
        dialect_embedding: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Seq2SeqLMOutput:
        """
        Args:
            input_features: Mel spectrogram [B, n_mels, seq_len]
            dialect_embedding: Pre-extracted dialect embedding [B, 768]
            decoder_input_ids: Decoder input token IDs [B, seq_len]
            labels: Target labels for loss computation [B, seq_len]
            attention_mask: Encoder attention mask [B, seq_len]
            decoder_attention_mask: Decoder attention mask [B, seq_len]
        """
        # Encode with dialect conditioning
        encoder_outputs = self.dialect_encoder(
            input_features=input_features,
            dialect_embedding=dialect_embedding,
            attention_mask=attention_mask,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Decode
        decoder_outputs = self.whisper.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        
        # Get logits
        lm_logits = self.whisper.proj_out(decoder_outputs.last_hidden_state)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return Seq2SeqLMOutput(loss=loss, logits=lm_logits)
    
    def generate(
        self,
        input_features: torch.Tensor,
        dialect_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Generate transcription with dialect conditioning."""
        encoder_outputs = self.dialect_encoder(
            input_features=input_features,
            dialect_embedding=dialect_embedding,
            attention_mask=attention_mask,
        )
        return self.whisper.generate(
            input_features=input_features,
            encoder_outputs=encoder_outputs,
            **kwargs
        )


def create_dialect_conditioned_whisper(
    whisper_model_name: str = "vasista22/whisper-tamil-medium",
    dialect_embedding_dim: int = 768
) -> DialectConditionedWhisper:
    """Create a dialect-conditioned Whisper model."""
    
    model = DialectConditionedWhisper(
        whisper_model_name=whisper_model_name,
        dialect_embedding_dim=dialect_embedding_dim
    )
    
    print(f"Created DialectConditionedWhisper:")
    print(f"  - Whisper model: {whisper_model_name}")
    print(f"  - Encoder hidden size: {model.encoder_hidden_size}")
    print(f"  - Dialect embedding dim: {dialect_embedding_dim}")
    print(f"  - Projection: {dialect_embedding_dim} → {model.encoder_hidden_size}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model
