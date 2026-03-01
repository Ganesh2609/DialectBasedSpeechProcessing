"""
Extract Dialect Embeddings from Trained Dialect Classifier

This script loads the best trained dialect classifier from:
    Dialect Classification/12_finetuned_learned_attentive_entire/train_data/checkpoints_1/best_model.pth

And extracts 768-dimensional dialect embeddings (output of learned attentive pooling,
BEFORE the classifier MLP) for all audio files used in ASR training.

Output: dialect_embeddings.npz
Format:
    - audio_ids: np.array of sample names
    - audio_paths: np.array of full paths
    - embeddings: np.array of shape [N, 768]
"""

import os
import sys
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

# Add the dialect classification folder to path so we can import the exact model
DIALECT_MODEL_PATH = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing/Dialect Classification/12_finetuned_learned_attentive_entire"
sys.path.insert(0, DIALECT_MODEL_PATH)

# Import the exact model architecture used during training
from model import TamilWav2VecLearnedAttentive, LearnedAttentivePooling


# ============================================================================
# Embedding Extraction
# ============================================================================

def load_dialect_classifier(checkpoint_path: str, device: str = 'cuda') -> TamilWav2VecLearnedAttentive:
    """Load trained dialect classifier from checkpoint."""
    
    # Initialize model with same architecture as training
    model = TamilWav2VecLearnedAttentive(
        model_name="Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
        num_classes=4,
        num_unfrozen_layers=4,
        classifier_hidden_size=512,
        classifier_dropout=0.3
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded dialect classifier from: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best F1 score: {checkpoint.get('best_metric', 'unknown'):.4f}")
    
    return model


def extract_embeddings(
    csv_path: str,
    checkpoint_path: str,
    output_path: str,
    device: str = 'cuda',
    batch_size: int = 1,
    max_length: int = 640000,
    sampling_rate: int = 16000
):
    """
    Extract dialect embeddings for all audio files.
    
    Args:
        csv_path: Path to transcripts.csv
        checkpoint_path: Path to dialect classifier checkpoint
        output_path: Path to save embeddings (.npz)
        device: Device to run on
        batch_size: Batch size for inference
        max_length: Maximum audio length in samples
        sampling_rate: Target sampling rate
    """
    
    # Load model
    model = load_dialect_classifier(checkpoint_path, device)
    
    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250"
    )
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} audio files to process")
    
    # Storage
    audio_ids = []
    audio_paths = []
    embeddings = []
    
    # Process each audio file
    with torch.no_grad():
        for idx in tqdm(range(len(df)), desc="Extracting embeddings"):
            row = df.iloc[idx]
            audio_path = row['audio_path']
            audio_id = row['name']
            
            try:
                # Load audio
                waveform, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
                
                # Truncate if too long
                if len(waveform) > max_length:
                    waveform = waveform[:max_length]
                
                # Process with feature extractor
                inputs = feature_extractor(
                    waveform,
                    sampling_rate=sampling_rate,
                    return_tensors='pt',
                    padding=False,
                    return_attention_mask=True,
                    do_normalize=False  # Audio already LUFS normalized
                )
                
                input_values = inputs['input_values'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                # Extract embedding using model components directly
                # Pass through wav2vec2 to get hidden states
                outputs = model.wav2vec2(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True
                )
                hidden_states = outputs.last_hidden_state  # [B, T', D]
                
                # Get reduced attention mask for the CNN-downsampled sequence
                if attention_mask is not None:
                    input_lengths = attention_mask.sum(dim=-1)
                    output_lengths = model.wav2vec2._get_feat_extract_output_lengths(input_lengths)
                    batch_size = attention_mask.size(0)
                    max_len = output_lengths.max().item()
                    output_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
                    for j, length in enumerate(output_lengths):
                        output_mask[j, :length] = 1
                else:
                    output_mask = None
                
                # Apply attentive pooling to get the embedding (BEFORE classifier)
                embedding = model.attentive_pooling(hidden_states, output_mask)  # [B, 768]
                
                # Store
                audio_ids.append(audio_id)
                audio_paths.append(audio_path)
                embeddings.append(embedding.cpu().numpy().squeeze())
                
            except Exception as e:
                print(f"\nError processing {audio_path}: {e}")
                continue
    
    # Convert to arrays
    audio_ids = np.array(audio_ids)
    audio_paths = np.array(audio_paths)
    embeddings = np.array(embeddings)
    
    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Save
    np.savez(
        output_path,
        audio_ids=audio_ids,
        audio_paths=audio_paths,
        embeddings=embeddings
    )
    
    print(f"Saved embeddings to: {output_path}")
    
    return audio_ids, audio_paths, embeddings


def main():
    # Paths
    base_path = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing"
    
    csv_path = f"{base_path}/Final Dataset/Train/transcripts.csv"
    checkpoint_path = f"{base_path}/Dialect Classification/12_finetuned_learned_attentive_entire/train_data/checkpoints_1/best_model.pth"
    output_path = f"{base_path}/Automatic Speech Recognition/dialect_embeddings.npz"
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Extract embeddings
    extract_embeddings(
        csv_path=csv_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        device=device,
        batch_size=1,
        max_length=640000,
        sampling_rate=16000
    )


if __name__ == '__main__':
    main()
