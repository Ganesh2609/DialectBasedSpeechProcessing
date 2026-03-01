"""
Training Script - Tamil Wav2Vec with Learned Attentive Pooling
Fine-tunes top 4 transformer layers + classifier
"""

import torch
from torch import nn
from model import TamilWav2VecLearnedAttentive
from trainer import DialectClassificationTrainer
from dataset import get_data_loaders
from transformers import Wav2Vec2FeatureExtractor


def main():
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Tamil Wav2Vec feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250"
    )

    # Model
    model = TamilWav2VecLearnedAttentive(
        model_name="Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
        num_classes=4,
        num_unfrozen_layers=4,
        classifier_hidden_size=512,
        classifier_dropout=0.3
    )

    # Dataset
    csv_path = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing/Final Dataset/Train/transcripts.csv"

    train_loader, val_loader = get_data_loaders(
        csv_path=csv_path,
        feature_extractor=feature_extractor,
        batch_size=4,  # Smaller batch for memory
        train_size=0.8,
        num_workers=12,
        max_length=640000
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Differential learning rates
    encoder_lr = 1e-5
    classifier_lr = 1e-4
    weight_decay = 1e-2

    param_groups = model.get_parameter_groups(
        encoder_lr=encoder_lr,
        classifier_lr=classifier_lr
    )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=True
    )

    loss_fn = nn.CrossEntropyLoss()

    # Paths
    base_path = "./train_data"

    trainer = DialectClassificationTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_classes=4,
        log_path=f'{base_path}/logs/training_1.log',
        num_epochs=20,
        checkpoint_path=f'{base_path}/checkpoints_1',
        graph_path=f'{base_path}/graphs/metrics_1.png',
        verbose=True,
        device=device
    )

    trainer.train()


if __name__ == '__main__':
    main()
