"""
Training Script for Dialect-Conditioned ASR (Cross-Attention)

This script trains a dialect-conditioned Whisper-Small model using
cross-attention conditioning instead of residual injection.

Staged Training:
- Phase 1 (warmup): Freeze Whisper, train projection only
- Phase 2 (fine-tune): Unfreeze decoder, keep encoder frozen

Usage:
    python training_whisper.py
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
    BASE_PATH = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing"
    CSV_PATH = f"{BASE_PATH}/Final Dataset/Train/transcripts.csv"
    EMBEDDINGS_PATH = f"{BASE_PATH}/ASR Whisper Small - Cross Attn/dialect_embeddings.npz"
    OUTPUT_PATH = f"{BASE_PATH}/ASR Whisper Small - Cross Attn/train_data"
    
    # Model - Using Whisper Small with cross-attention conditioning
    WHISPER_MODEL = "vasista22/whisper-tamil-small"
    DIALECT_EMBEDDING_DIM = 768
    
    # Training hyperparameters
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-2
    NUM_EPOCHS = 20
    PROJECTION_WARMUP_EPOCHS = 4  # Warm up projection before fine-tuning
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
    
    print("Creating dialect-conditioned Whisper model (cross-attention)...")
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
        log_path=f'{OUTPUT_PATH}/logs/training_1.log',
        num_epochs=NUM_EPOCHS,
        projection_warmup_epochs=PROJECTION_WARMUP_EPOCHS,
        checkpoint_path=f'{OUTPUT_PATH}/checkpoints/training 1',
        graph_path=f'{OUTPUT_PATH}/graphs/metrics_1.png',
        verbose=True,
        device=device
    )
    
    # =========================================================================
    # Train
    # =========================================================================
    
    print("\nStarting training...")
    print(f"Phase 1 (Warmup): Epochs 1-{PROJECTION_WARMUP_EPOCHS} - Train projection only")
    print(f"Phase 2 (Fine-tune): Epochs {PROJECTION_WARMUP_EPOCHS + 1}-{NUM_EPOCHS} - Train decoder + projection")
    print("=" * 60)
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved to: {OUTPUT_PATH}/checkpoints/best_model.pth")
    print(f"Graphs saved to: {OUTPUT_PATH}/graphs/metrics_1.png")


if __name__ == '__main__':
    main()
