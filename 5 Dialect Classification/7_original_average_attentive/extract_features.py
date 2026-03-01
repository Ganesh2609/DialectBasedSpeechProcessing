"""
Feature Extraction Script for Dialect Classification
Model: facebook/wav2vec2-large-xlsr-53
Pooling: average_attentive (both mean and attentive pooling)
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

MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
POOLING_TYPE = "average_attentive"
SAMPLING_RATE = 16000
MAX_LENGTH = 640000

CSV_PATH = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing/Final Dataset/Train/transcripts.csv"
OUTPUT_PATH = "./features.npz"

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

def extract_features(model, feature_extractor, waveform, device, pooling_type="average_attentive"):
    inputs = feature_extractor(waveform, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True, do_normalize=False)
    input_values = inputs["input_values"].to(device)
    
    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state
    
    if pooling_type == "average":
        pooled = hidden_states.mean(dim=1)
    elif pooling_type == "attentive":
        norms = torch.norm(hidden_states, dim=-1, keepdim=True)
        weights = torch.softmax(norms, dim=1)
        pooled = (hidden_states * weights).sum(dim=1)
    elif pooling_type == "average_attentive":
        mean_pooled = hidden_states.mean(dim=1)
        norms = torch.norm(hidden_states, dim=-1, keepdim=True)
        weights = torch.softmax(norms, dim=1)
        attentive_pooled = (hidden_states * weights).sum(dim=1)
        pooled = torch.cat([mean_pooled, attentive_pooled], dim=-1)
    
    return pooled.cpu().numpy().squeeze(0)

def main():
    print(f"Model: {MODEL_NAME}\nPooling: {POOLING_TYPE}")
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
    
    names, audio_paths, embeddings, labels = [], [], [], []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = row['audio_path']
        try:
            waveform = load_audio(audio_path)
            embedding = extract_features(model, feature_extractor, waveform, device, POOLING_TYPE)
            names.append(row['name'])
            audio_paths.append(audio_path)
            embeddings.append(embedding)
            labels.append(DIALECT_LABELS[extract_dialect(audio_path)])
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            continue
    
    embeddings = np.stack(embeddings, axis=0)
    labels = np.array(labels)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    np.savez(OUTPUT_PATH, names=np.array(names), audio_paths=np.array(audio_paths), embeddings=embeddings, labels=labels)
    print(f"Saved features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
