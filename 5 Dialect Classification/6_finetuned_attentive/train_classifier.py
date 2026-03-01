"""
Training Script for Dialect Classification
Uses pre-extracted features from features.npz

Model: facebook/wav2vec2-large-xlsr-53
Pooling: average (mean pooling only)
Input dim: 1024
"""

import torch
from torch import nn
from model import DialectClassifier
from trainer import DialectClassificationTrainer
from dataset import get_data_loaders


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration
    INPUT_DIM = 768  # 768 for Tamil Wav2Vec (average or attentive)
    BATCH_SIZE = 32
    NUM_EPOCHS = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Load pre-extracted features
    train_loader, val_loader = get_data_loaders(
        npz_path="./features.npz",
        batch_size=BATCH_SIZE,
        train_size=0.8
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = DialectClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=512,
        num_classes=4,
        dropout=0.3
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )

    # Loss
    loss_fn = nn.CrossEntropyLoss()

    # Trainer
    trainer = DialectClassificationTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_classes=4,
        log_path='./train_data/logs/training_1.log',
        num_epochs=NUM_EPOCHS,
        checkpoint_path='./train_data/checkpoints/train_1',
        graph_path='./train_data/graphs/metrics_1.png',
        verbose=True,
        device=device
    )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
