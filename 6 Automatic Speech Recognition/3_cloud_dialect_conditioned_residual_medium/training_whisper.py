"""
Training Script for Dialect-Conditioned ASR

This script trains a dialect-conditioned Whisper model on Tamil speech data.
Dialect embeddings are pre-extracted and injected into the encoder.

Usage:
    1. First run: python extract_dialect_embeddings.py
    2. Then run: python training_whisper.py
"""

import torch
from model import create_dialect_conditioned_whisper
from trainer import DialectConditionedASRTrainer
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
    BASE_PATH = "Cloud ASR Training"
    CSV_PATH = f"{BASE_PATH}/../Final Dataset/Train/transcripts.csv"
    EMBEDDINGS_PATH = f"{BASE_PATH}/dialect_embeddings.npz"
    OUTPUT_PATH = f"{BASE_PATH}/train_data"
    
    # Audio path conversion for cloud (converts Windows paths in CSV to relative paths)
    # Old: E:/Work/My Papers/.../Final Dataset/Train/...
    # New: {AUDIO_BASE_PATH}/Final Dataset/Train/...
    AUDIO_BASE_PATH = f"{BASE_PATH}/.."
    
    # Model
    WHISPER_MODEL = "vasista22/whisper-tamil-medium"
    DIALECT_EMBEDDING_DIM = 768
    
    # Training hyperparameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-2
    NUM_EPOCHS = 10
    MAX_AUDIO_LENGTH = 640000  # 40 seconds at 16kHz
    NUM_WORKERS = 7
    
    # =========================================================================
    # Load Processor
    # =========================================================================
    
    print(f"Loading Whisper processor from: {WHISPER_MODEL}")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, task="transcribe")
    
    # =========================================================================
    # Create Model
    # =========================================================================
    
    print("Creating dialect-conditioned Whisper model...")
    model = create_dialect_conditioned_whisper(
        whisper_model_name=WHISPER_MODEL,
        dialect_embedding_dim=DIALECT_EMBEDDING_DIM
    )
    
    # =========================================================================
    # Load Data
    # =========================================================================
    
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        csv_path=CSV_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        processor=processor,
        audio_base_path=AUDIO_BASE_PATH,
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
    
    trainer = DialectConditionedASRTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        processor=processor,
        log_path=f'{OUTPUT_PATH}/logs/training_2.log',
        num_epochs=NUM_EPOCHS,
        checkpoint_path=f'{OUTPUT_PATH}/checkpoints/training 2',
        graph_path=f'{OUTPUT_PATH}/graphs/training_2.png',
        verbose=True,
        device=device
    )
    
    # =========================================================================
    # Train
    # =========================================================================
    
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved to: {OUTPUT_PATH}/checkpoints/best_model.pth")
    print(f"Graphs saved to: {OUTPUT_PATH}/graphs/metrics_1.png")


if __name__ == '__main__':
    main()
