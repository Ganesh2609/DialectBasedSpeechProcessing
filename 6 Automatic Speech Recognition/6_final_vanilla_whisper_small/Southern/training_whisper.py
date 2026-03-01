"""
Training Script for Srilankan Dialect - Whisper Small
"""

import sys
sys.path.append('..')

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from trainer import ASRTrainer
from dataset import get_data_loaders


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    BASE_PATH = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing"
    CSV_PATH = f"{BASE_PATH}/Final Dataset/Train/transcripts.csv"
    OUTPUT_PATH = f"{BASE_PATH}/Final Whisper Training - Small/Srilankan/train_data"
    
    WHISPER_MODEL = "vasista22/whisper-tamil-small"
    DIALECT = "Srilankan"
    
    BATCH_SIZE = 2 
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-2
    NUM_EPOCHS = 50
    MAX_AUDIO_LENGTH = 640000
    NUM_WORKERS = 12
    
    print(f"Loading from: {WHISPER_MODEL}")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
    
    print(f"Loading data (dialect: {DIALECT})...")
    train_loader, val_loader = get_data_loaders(
        csv_path=CSV_PATH,
        processor=processor,
        dialect=DIALECT,
        batch_size=BATCH_SIZE,
        train_size=0.8,
        num_workers=NUM_WORKERS,
        max_audio_length=MAX_AUDIO_LENGTH,
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7, verbose=True)
    
    trainer = ASRTrainer(
        model=model, train_loader=train_loader, test_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, processor=processor,
        log_path=f'{OUTPUT_PATH}/logs/training.log', num_epochs=NUM_EPOCHS,
        checkpoint_path=f'{OUTPUT_PATH}/checkpoints', graph_path=f'{OUTPUT_PATH}/graphs/training.png',
        verbose=True, device=device
    )
    
    print(f"\nStarting training ({DIALECT} dialect)...")
    trainer.train()
    print(f"Best model saved to: {OUTPUT_PATH}/checkpoints/best_model.pth")


if __name__ == '__main__':
    main()
