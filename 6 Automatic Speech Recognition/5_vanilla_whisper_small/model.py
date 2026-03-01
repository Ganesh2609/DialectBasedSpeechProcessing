"""
Vanilla Whisper ASR Model (Small) - No Dialect Conditioning

Standard Whisper model without any dialect embedding injection.
Used as baseline for comparison with dialect-conditioned variants.

Model: vasista22/whisper-tamil-small
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


class VanillaWhisper(nn.Module):
    """
    Standard Whisper ASR without dialect conditioning.
    Thin wrapper around WhisperForConditionalGeneration for consistent interface.
    """
    
    def __init__(
        self,
        whisper_model_name: str = "vasista22/whisper-tamil-small",
    ):
        super().__init__()
        
        # Load pretrained Whisper
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        self.config = self.whisper.config
    
    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Seq2SeqLMOutput:
        """
        Args:
            input_features: Mel spectrogram [B, n_mels, seq_len]
            decoder_input_ids: Decoder input token IDs [B, seq_len]
            labels: Target labels for loss computation [B, seq_len]
            attention_mask: Encoder attention mask [B, seq_len]
            decoder_attention_mask: Decoder attention mask [B, seq_len]
        """
        # Standard Whisper forward pass
        outputs = self.whisper(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        
        return outputs
    
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Generate transcription using standard Whisper."""
        return self.whisper.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs
        )


def create_vanilla_whisper(
    whisper_model_name: str = "vasista22/whisper-tamil-small",
) -> VanillaWhisper:
    """Create a vanilla Whisper model without dialect conditioning."""
    
    model = VanillaWhisper(
        whisper_model_name=whisper_model_name,
    )
    
    print(f"Created VanillaWhisper:")
    print(f"  - Whisper model: {whisper_model_name}")
    print(f"  - No dialect conditioning")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model
