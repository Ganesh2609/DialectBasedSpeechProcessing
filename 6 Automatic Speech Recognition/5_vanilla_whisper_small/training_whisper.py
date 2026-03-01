"""
Training Script for Standard ASR (Whisper Small) - No Dialect Conditioning

This script trains a vanilla Whisper-Small model on Tamil speech data.
No dialect embeddings are used - baseline comparison model.

Usage:
    python training_whisper.py
"""

import torch
from model import create_vanilla_whisper
from trainer import ASRTrainer
from dataset import get_data_loaders
from transformers import WhisperProcessor


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    BASE_PATH = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing"
    CSV_PATH = f"{BASE_PATH}/Final Dataset/Train/transcripts.csv"
    OUTPUT_PATH = f"{BASE_PATH}/ASR Whisper Small - No dialect/train_data"
    
    # Model - Using Whisper Small (no dialect conditioning)
    WHISPER_MODEL = "vasista22/whisper-tamil-small"
    
    # Training hyperparameters
    BATCH_SIZE = 2 
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-2
    NUM_EPOCHS = 32
    MAX_AUDIO_LENGTH = 640000  # 40 seconds at 16kHz
    NUM_WORKERS = 12
    
    # =========================================================================
    # Load Processor
    # =========================================================================
    
    print(f"Loading Whisper processor from: {WHISPER_MODEL}")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, task="transcribe")
    
    # =========================================================================
    # Create Model
    # =========================================================================
    
    print("Creating vanilla Whisper model (no dialect conditioning)...")
    model = create_vanilla_whisper(
        whisper_model_name=WHISPER_MODEL,
    )
    
    # =========================================================================
    # Load Data
    # =========================================================================
    
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        csv_path=CSV_PATH,
        processor=processor,
        batch_size=BATCH_SIZE,
        train_size=0.8,
        num_workers=NUM_WORKERS,
        max_audio_length=MAX_AUDIO_LENGTH,
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # =========================================================================
    # Setup Training
    # =========================================================================
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler (reduce on plateau based on WER)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",  # Lower WER is better
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=True
    )
    
    # =========================================================================
    # Create Trainer
    # =========================================================================
    
    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        processor=processor,
        log_path=f'{OUTPUT_PATH}/logs/training_1.log',
        num_epochs=NUM_EPOCHS,
        checkpoint_path=f'{OUTPUT_PATH}/checkpoints/training 1',
        graph_path=f'{OUTPUT_PATH}/graphs/training_1.png',
        verbose=True,
        device=device
    )
    
    # =========================================================================
    # Train
    # =========================================================================
    
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()
    # To resume: trainer.train(resume_from=f'{OUTPUT_PATH}/checkpoints/training 1/model_epoch_X.pth')
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved to: {OUTPUT_PATH}/checkpoints/best_model.pth")
    print(f"Graphs saved to: {OUTPUT_PATH}/graphs/training_1.png")


if __name__ == '__main__':
    main()
