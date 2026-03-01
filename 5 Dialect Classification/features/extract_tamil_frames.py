"""
Extract RAW frame-level features from Wav2Vec2 (no pooling).
Model: Harveenchadha/vakyansh-wav2vec2-tamil-tam-250
Output: [N, max_T, 768] with lengths for masking
"""

import os
import warnings
warnings.filterwarnings("ignore")

import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250"
SAMPLING_RATE = 16000
MAX_LENGTH = 640000  # 40 seconds

CSV_PATH = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing/Final Dataset/Train/transcripts.csv"
OUTPUT_PATH = "./tamil_wav2vec_frames.npz"

DIALECT_LABELS = {'Central_Dialect': 0, 'Northern_Dialect': 1, 'Southern_Dialect': 2, 'Western_Dialect': 3}


def extract_dialect(audio_path: str) -> str:
    path_parts = audio_path.replace('\\', '/').split('/')
    for part in path_parts:
        if part in DIALECT_LABELS:
            return part
    raise ValueError(f"Could not extract dialect from path: {audio_path}")


def load_audio(audio_path: str) -> np.ndarray:
    waveform, sr = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    if len(waveform) > MAX_LENGTH:
        waveform = waveform[:MAX_LENGTH]
    return waveform


def main():
    print(f"Model: {MODEL_NAME}")
    print("Extracting RAW frame-level features (no pooling)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Loading model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    
    df = pd.read_csv(CSV_PATH)
    print(f"Total samples: {len(df)}")
    
    names = []
    audio_paths = []
    all_embeddings = []
    lengths = []
    labels = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = row['audio_path']
        try:
            waveform = load_audio(audio_path)
            
            inputs = feature_extractor(
                waveform, sampling_rate=SAMPLING_RATE, 
                return_tensors="pt", padding=True, do_normalize=False
            )
            input_values = inputs["input_values"].to(device)
            
            with torch.no_grad():
                outputs = model(input_values)
                hidden_states = outputs.last_hidden_state  # [1, T, 768]
            
            frames = hidden_states.cpu().numpy().squeeze(0)  # [T, 768]
            
            names.append(row['name'])
            audio_paths.append(audio_path)
            all_embeddings.append(frames)
            lengths.append(frames.shape[0])
            labels.append(DIALECT_LABELS[extract_dialect(audio_path)])
            
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            continue
    
    # Pad to max length
    max_len = max(lengths)
    hidden_dim = all_embeddings[0].shape[1]
    print(f"\nMax sequence length: {max_len}, Hidden dim: {hidden_dim}")
    
    padded_embeddings = np.zeros((len(all_embeddings), max_len, hidden_dim), dtype=np.float32)
    for i, emb in enumerate(all_embeddings):
        padded_embeddings[i, :emb.shape[0], :] = emb
    
    lengths = np.array(lengths, dtype=np.int32)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"Embeddings shape: {padded_embeddings.shape}")
    print(f"Lengths shape: {lengths.shape}")
    
    np.savez_compressed(
        OUTPUT_PATH,
        names=np.array(names),
        audio_paths=np.array(audio_paths),
        embeddings=padded_embeddings,
        lengths=lengths,
        labels=labels
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
