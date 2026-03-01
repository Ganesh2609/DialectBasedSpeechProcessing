"""
Training Script - Learned Average + Attentive Pooling
Model: Tamil Wav2Vec (1536 combined dim)
"""

import torch
from torch import nn
from model import DialectClassifierLearnedAverageAttentive
from trainer import DialectClassificationTrainer
from dataset import get_data_loaders


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    INPUT_DIM = 768  # Tamil Wav2Vec (output = 1536)
    BATCH_SIZE = 32
    NUM_EPOCHS = 256

    train_loader, val_loader = get_data_loaders(
        npz_path="../features/tamil_wav2vec_frames.npz",
        batch_size=BATCH_SIZE,
        train_size=0.8
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = DialectClassifierLearnedAverageAttentive(
        input_dim=INPUT_DIM, hidden_dim=512, num_classes=4, dropout=0.3
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )

    trainer = DialectClassificationTrainer(
        model=model, 
        train_loader=train_loader, 
        test_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(), 
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
    trainer.train()


if __name__ == '__main__':
    main()
