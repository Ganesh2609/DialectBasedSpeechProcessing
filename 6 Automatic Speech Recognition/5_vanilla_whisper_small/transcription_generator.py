"""
Transcription Generator - Single Sample Inference (No Dialect)

Generates transcriptions for validation set one sample at a time.
Uses Whisper's default optimized generate method.

Output: transcriptions_comparison.csv with columns:
- audio_id
- original_transcription  
- predicted_transcription
"""

import os
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from torch.utils.data import random_split
from transformers import WhisperProcessor
from model import create_vanilla_whisper


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    
    BASE_PATH = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing"
    CSV_PATH = f"{BASE_PATH}/Final Dataset/Train/transcripts.csv"
    CHECKPOINT_PATH = f"{BASE_PATH}/ASR Whisper Small - No dialect/train_data/checkpoints/training 1/best_model.pth"
    OUTPUT_PATH = f"{BASE_PATH}/ASR Whisper Small - No dialect/train_data/transcriptions_comparison_1.csv"
    
    WHISPER_MODEL = "vasista22/whisper-tamil-small"
    SAMPLING_RATE = 16000
    RANDOM_SEED = 17
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # =========================================================================
    # Load Model
    # =========================================================================
    
    print(f"Loading processor from: {WHISPER_MODEL}")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, task="transcribe")
    
    print("Creating model...")
    model = create_vanilla_whisper(whisper_model_name=WHISPER_MODEL)
    
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best WER: {checkpoint.get('best_metric', 'unknown'):.2f}%")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    # Get validation split indices (same seed as training)
    indices = list(range(len(df)))
    train_len = int(0.8 * len(indices))
    val_len = len(indices) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, val_indices = random_split(indices, [train_len, val_len], generator=generator)
    val_indices = list(val_indices)
    
    print(f"Validation samples: {len(val_indices)}")
    
    # Limit to 100 samples
    val_indices = val_indices[:100]
    print(f"Using {len(val_indices)} samples for generation")
    
    # =========================================================================
    # Generate Transcriptions
    # =========================================================================
    
    print("\nGenerating transcriptions...")
    results = []
    
    with torch.no_grad():
        for idx in tqdm(val_indices, desc="Generating"):
            row = df.iloc[idx]
            audio_path = row['audio_path']
            audio_id = row['name']
            original_text = str(row['transcription'])
            
            # Load audio
            waveform, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
            
            # Process audio - pad to Whisper's expected 30 seconds (3000 mel frames)
            inputs = processor.feature_extractor(
                waveform, 
                sampling_rate=SAMPLING_RATE, 
                return_tensors='pt', 
                padding='longest',
                max_length=480000,  # 30 seconds at 16kHz
                truncation=True,
            )
            input_features = inputs['input_features'].to(DEVICE)
            
            # Generate (no dialect embedding needed)
            generated_ids = model.generate(
                input_features=input_features,
            )
            
            predicted_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            results.append({
                'audio_id': audio_id,
                'original_transcription': original_text,
                'predicted_transcription': predicted_text,
            })
    
    print(f"\nGenerated {len(results)} transcriptions")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"Saved to: {OUTPUT_PATH}")
    
    # Show sample
    print("\nSample results:")
    print(df_results.head(5).to_string())


if __name__ == '__main__':
    main()
