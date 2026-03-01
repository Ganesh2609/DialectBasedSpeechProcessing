"""
Transcription Generator for Whisper Small Models
Uses WhisperForConditionalGeneration directly with repetition penalty.
"""

import os
import sys
sys.path.append('..')

import torch
import librosa
import pandas as pd
from tqdm import tqdm
from torch.utils.data import random_split
from transformers import WhisperForConditionalGeneration, WhisperProcessor

RANDOM_SEED = 17


def generate_transcriptions(
    checkpoint_path: str,
    output_path: str,
    csv_path: str,
    whisper_model: str = "vasista22/whisper-tamil-small",
    dialect: str = None,
    num_samples: int = 100,
):
    """Generate transcriptions for validation set with repetition penalty."""
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    print(f"Loading from: {whisper_model}")
    processor = WhisperProcessor.from_pretrained(whisper_model, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best WER: {checkpoint.get('best_metric', 'unknown'):.2f}%")
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    if dialect is not None:
        df = df[df['dialect'] == dialect].reset_index(drop=True)
        print(f"Filtered to dialect '{dialect}': {len(df)} samples")
    
    indices = list(range(len(df)))
    train_len = int(0.8 * len(indices))
    val_len = len(indices) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, val_indices = random_split(indices, [train_len, val_len], generator=generator)
    val_indices = list(val_indices)
    
    print(f"Validation samples: {len(val_indices)}")
    val_indices = val_indices[:num_samples]
    print(f"Using {len(val_indices)} samples for generation")
    
    print("\nGenerating transcriptions...")
    results = []
    
    with torch.no_grad():
        for idx in tqdm(val_indices, desc="Generating"):
            row = df.iloc[idx]
            audio_path = row['audio_path']
            audio_id = row['name']
            original_text = str(row['transcription'])
            
            waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
            
            inputs = processor.feature_extractor(
                waveform, sampling_rate=16000, return_tensors='pt', padding='max_length',
            )
            input_features = inputs['input_features'].to(DEVICE)
            
            # Generate with repetition penalty to prevent loops
            generated_ids = model.generate(
                input_features=input_features,
                temperature=0,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
            )
            predicted_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            results.append({
                'audio_id': audio_id,
                'original_transcription': original_text,
                'predicted_transcription': predicted_text,
            })
    
    print(f"\nGenerated {len(results)} transcriptions")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved to: {output_path}")
    
    print("\nSample results:")
    print(df_results.head(5).to_string())


if __name__ == '__main__':
    BASE_PATH = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing"
    
    generate_transcriptions(
        checkpoint_path=f"{BASE_PATH}/Final Whisper Training - Small/Unified/train_data/checkpoints/best_model.pth",
        output_path=f"{BASE_PATH}/Final Whisper Training - Small/Unified/train_data/transcriptions.csv",
        csv_path=f"{BASE_PATH}/Final Dataset/Train/transcripts.csv",
        whisper_model="vasista22/whisper-tamil-small",
        dialect=None,
        num_samples=100,
    )
