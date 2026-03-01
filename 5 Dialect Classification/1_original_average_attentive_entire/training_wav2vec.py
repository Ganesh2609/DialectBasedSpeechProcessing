import torch
from torch import nn
from model import Wav2Vec2ForDialectClassification
from trainer import DialectClassificationTrainer
from dataset import get_data_loaders
from transformers import Wav2Vec2FeatureExtractor


def main():

    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53"
    )

    # Initialize model
    model = Wav2Vec2ForDialectClassification(
        model_name="facebook/wav2vec2-large-xlsr-53",
        num_classes=4,
        num_unfrozen_layers=4,
        classifier_hidden_size=512,
        classifier_dropout=0.3
    )

    # Dataset path
    csv_path = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing/Final Dataset/Train/transcripts.csv"

    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        csv_path=csv_path,
        feature_extractor=feature_extractor,
        batch_size=2,
        train_size=0.8,
        num_workers=12,
        max_length=640000  # 40 seconds at 16kHz
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Hyperparameters
    encoder_lr = 1e-5
    classifier_lr = 1e-4
    weight_decay = 1e-2

    # Get parameter groups with differential learning rates
    param_groups = model.get_parameter_groups(
        encoder_lr=encoder_lr,
        classifier_lr=classifier_lr
    )

    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",  # Maximize F1 score
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=True
    )

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Output paths
    base_path = "E:/Work/My Papers/Dravidian Lang Tech 2026/Dialect based speech processing/Dialect Classification/train_data"

    # Trainer
    trainer = DialectClassificationTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_classes=4,
        log_path=f'{base_path}/logs/dialect_classification_1.log',
        num_epochs=16,
        checkpoint_path=f'{base_path}/checkpoints/dialect_classification_1',
        graph_path=f'{base_path}/graphs/dialect_classification_1.png',
        verbose=True,
        device=device
    )

    # Start training
    trainer.train()
    # To resume from checkpoint, use:
    # trainer.train(resume_from=f'{base_path}/checkpoints/dialect_classification_1/model_epoch_5.pth')


if __name__ == '__main__':
    main()
