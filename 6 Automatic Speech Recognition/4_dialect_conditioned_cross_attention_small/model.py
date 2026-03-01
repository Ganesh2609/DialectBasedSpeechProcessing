"""
Dialect-Conditioned Whisper ASR Model (Cross-Attention)

New approach: Dialect embedding as a context token for decoder cross-attention.

Architecture:
1. Run Whisper encoder normally (no dialect modification)
2. Project 768-dim dialect embedding to single token of Whisper hidden size
3. Concatenate dialect token in front of encoder outputs
4. Decoder cross-attention attends to [dialect_token, encoder_outputs]

Model: vasista22/whisper-tamil-small
Dialect Embedding: 768-dim (frozen, from wav2vec2 Tamil classifier)
Whisper Hidden Size: 768
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput


class DialectConditionedWhisperCrossAttn(nn.Module):
    """
    Whisper ASR with dialect conditioning via cross-attention.
    
    Dialect embedding is projected to a single token and concatenated
    to encoder outputs. Decoder cross-attention attends to both.
    """
    
    def __init__(
        self,
        whisper_model_name: str = "vasista22/whisper-tamil-small",
        dialect_embedding_dim: int = 768,
    ):
        super().__init__()
        
        # Load pretrained Whisper
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        self.config = self.whisper.config
        self.encoder_hidden_size = self.config.d_model  # 768 for small
        
        # Dialect projection: 768 → single token of hidden_size
        # Simple linear projection with layer norm for stability
        self.dialect_projection = nn.Sequential(
            nn.Linear(dialect_embedding_dim, self.encoder_hidden_size),
            nn.LayerNorm(self.encoder_hidden_size),
        )
        
        # Track training phase for staged training
        self._warmup_mode = True
    
    def set_warmup_mode(self, warmup: bool):
        """Set whether we're in warmup mode (freeze Whisper) or not."""
        self._warmup_mode = warmup
        
        if warmup:
            # Freeze entire Whisper model
            for param in self.whisper.parameters():
                param.requires_grad = False
            # Only projection is trainable
            for param in self.dialect_projection.parameters():
                param.requires_grad = True
            print("Warmup mode: Whisper frozen, training projection only")
        else:
            # Unfreeze decoder, keep encoder frozen
            for param in self.whisper.model.encoder.parameters():
                param.requires_grad = False
            for param in self.whisper.model.decoder.parameters():
                param.requires_grad = True
            for param in self.whisper.proj_out.parameters():
                param.requires_grad = True
            for param in self.dialect_projection.parameters():
                param.requires_grad = True
            print("Fine-tune mode: Encoder frozen, Decoder + Projection trainable")
    
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
            attention_mask: Encoder attention mask (unused, Whisper doesn't use it)
            decoder_attention_mask: Decoder attention mask
        """
        # Step 1: Run Whisper encoder normally (no dialect modification)
        encoder_outputs = self.whisper.model.encoder(
            input_features=input_features,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [B, T, 768]
        
        # Step 2: Project dialect embedding to single token
        # [B, 768] → [B, 768] → [B, 1, 768]
        dialect_token = self.dialect_projection(dialect_embedding).unsqueeze(1)
        
        # Step 3: Concatenate dialect token in front of encoder outputs
        # [B, 1, 768] + [B, T, 768] → [B, 1+T, 768]
        combined_hidden_states = torch.cat([dialect_token, encoder_hidden_states], dim=1)
        
        # Step 4: Pass combined sequence to decoder
        decoder_outputs = self.whisper.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=combined_hidden_states,
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
        **kwargs
    ):
        """Generate transcription with dialect conditioning."""
        # Run encoder
        encoder_outputs = self.whisper.model.encoder(input_features=input_features)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Project and concatenate dialect token
        dialect_token = self.dialect_projection(dialect_embedding).unsqueeze(1)
        combined_hidden_states = torch.cat([dialect_token, encoder_hidden_states], dim=1)
        
        # Create modified encoder outputs for generation
        modified_encoder_outputs = BaseModelOutput(last_hidden_state=combined_hidden_states)
        
        return self.whisper.generate(
            input_features=input_features,
            encoder_outputs=modified_encoder_outputs,
            **kwargs
        )


def create_dialect_conditioned_whisper(
    whisper_model_name: str = "vasista22/whisper-tamil-small",
    dialect_embedding_dim: int = 768
) -> DialectConditionedWhisperCrossAttn:
    """Create a dialect-conditioned Whisper model with cross-attention conditioning."""
    
    model = DialectConditionedWhisperCrossAttn(
        whisper_model_name=whisper_model_name,
        dialect_embedding_dim=dialect_embedding_dim
    )
    
    print(f"Created DialectConditionedWhisperCrossAttn:")
    print(f"  - Whisper model: {whisper_model_name}")
    print(f"  - Encoder hidden size: {model.encoder_hidden_size}")
    print(f"  - Dialect embedding dim: {dialect_embedding_dim}")
    print(f"  - Conditioning: Cross-attention (dialect token + encoder outputs)")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model
