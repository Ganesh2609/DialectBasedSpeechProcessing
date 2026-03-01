"""
Trainer for Standard ASR (No Dialect Conditioning)

Key features:
- Greedy decoding ONLY during training and validation
- Tracks train/val loss and WER (Word Error Rate)
- Best model selected by lowest validation WER
- 2x2 plot grid: Train Loss, Val Loss, Train WER, Val WER
"""

import os
import torch 
from torch import nn
import matplotlib.pyplot as plt
from typing import Optional
from logger import TrainingLogger
from tqdm import tqdm
from torchmetrics.text import WordErrorRate


class ASRTrainer:
    """
    Trainer for standard Whisper ASR (no dialect conditioning).
    
    Uses greedy decoding for training and validation.
    Best model selected by lowest validation WER.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader, 
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        processor = None,
        log_path: Optional[str] = './logs/training.log',
        num_epochs: Optional[int] = 20,
        checkpoint_path: Optional[str] = './checkpoints',
        graph_path: Optional[str] = './graphs/metrics.png',
        verbose: Optional[bool] = True,
        device: Optional[torch.device] = None
    ) -> None:
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        
        self.logger = TrainingLogger(log_path=log_path)

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.processor = processor

        self.loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optimizer or torch.optim.AdamW(params=self.model.parameters(), lr=1e-5)
        self.scheduler = scheduler

        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.graph_path = graph_path
        self.verbose = verbose
        self.loss_update_step = 50

        self.current_epoch = 1
        self.current_step = 1
        self.best_metric = float('inf')  # Best WER (lower is better)

        # WER metric
        self.wer_metric = WordErrorRate()

        self.history = {
            'Training Loss': [],
            'Training WER': [],
            'Validation Loss': [],
            'Validation WER': [],
        }

        self.step_history = {
            'Training Loss': [],
            'Training WER': [],
            'Validation Loss': [],
            'Validation WER': [],
        }


    def update_plot(self) -> None:
        """Generate 2x2 plot of training and validation metrics."""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Row 1: Loss metrics
        axs[0, 0].plot(self.step_history['Training Loss'], color='blue', label='Training Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        axs[0, 1].plot(self.step_history['Validation Loss'], color='green', label='Validation Loss')
        axs[0, 1].set_title('Validation Loss')
        axs[0, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)

        # Row 2: WER metrics
        axs[1, 0].plot(self.step_history['Training WER'], color='red', label='Training WER')
        axs[1, 0].set_title('Training Word Error Rate')
        axs[1, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 0].set_ylabel('WER (%)')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        axs[1, 1].plot(self.step_history['Validation WER'], color='purple', label='Validation WER')
        axs[1, 1].set_title('Validation Word Error Rate')
        axs[1, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 1].set_ylabel('WER (%)')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.graph_path, dpi=150)
        plt.close(fig)

        return


    def train_epoch(self) -> None:
        """Train for one epoch using greedy decoding."""
        
        self.model.train()
        total_loss = 0.0
        total_wer = 0.0

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                  desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Training)') as t:
            
            for i, batch in t:
                
                input_features = batch['input_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                transcription_text = batch['transcription_text']
                
                # Create decoder_input_ids by shifting labels right
                # Whisper uses decoder_start_token_id (usually <|startoftranscript|> = 50258)
                decoder_input_ids = labels.clone()
                decoder_input_ids[decoder_input_ids == -100] = self.processor.tokenizer.pad_token_id
                # Shift right: prepend decoder_start_token and remove last token
                decoder_start_token_id = self.model.whisper.config.decoder_start_token_id
                decoder_input_ids = torch.cat([
                    torch.full((labels.size(0), 1), decoder_start_token_id, dtype=labels.dtype, device=self.device),
                    decoder_input_ids[:, :-1]
                ], dim=1)

                # Forward pass
                outputs = self.model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                )
                
                loss = outputs.loss
                logits = outputs.logits

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Greedy decoding for WER computation
                with torch.no_grad():
                    predicted_ids = torch.argmax(logits, dim=-1)
                    predictions = self.processor.tokenizer.batch_decode(
                        predicted_ids, 
                        skip_special_tokens=True
                    )
                    wer = self.wer_metric(predictions, transcription_text).item() * 100

                total_loss += loss.item()
                total_wer += wer

                self.current_step += 1

                t.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss / (i + 1):.4f}',
                    'WER': f'{total_wer / (i + 1):.2f}%',
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Training Loss'].append(total_loss / (i + 1))
                    self.step_history['Training WER'].append(total_wer / (i + 1))
                    self.update_plot()

        train_loss = total_loss / len(self.train_loader)
        train_wer = total_wer / len(self.train_loader)

        self.history['Training Loss'].append(train_loss)
        self.history['Training WER'].append(train_wer)
        
        self.logger.info(f"Training loss for epoch {self.current_epoch}: {train_loss:.4f}")
        self.logger.info(f"Training WER for epoch {self.current_epoch}: {train_wer:.2f}%\n")

        return


    def test_epoch(self) -> None:
        """Validate for one epoch using greedy decoding."""
        
        self.model.eval()
        total_loss = 0.0
        total_wer = 0.0

        with tqdm(enumerate(self.test_loader), total=len(self.test_loader), 
                  desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Validation)') as t:
            
            for i, batch in t:
                
                input_features = batch['input_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                transcription_text = batch['transcription_text']
                
                with torch.no_grad():
                    # Create decoder_input_ids by shifting labels right
                    decoder_input_ids = labels.clone()
                    decoder_input_ids[decoder_input_ids == -100] = self.processor.tokenizer.pad_token_id
                    decoder_start_token_id = self.model.whisper.config.decoder_start_token_id
                    decoder_input_ids = torch.cat([
                        torch.full((labels.size(0), 1), decoder_start_token_id, dtype=labels.dtype, device=self.device),
                        decoder_input_ids[:, :-1]
                    ], dim=1)
                    
                    # Forward pass
                    outputs = self.model(
                        input_features=input_features,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels,
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits

                    # Greedy decoding for WER computation
                    predicted_ids = torch.argmax(logits, dim=-1)
                    predictions = self.processor.tokenizer.batch_decode(
                        predicted_ids, 
                        skip_special_tokens=True
                    )
                    wer = self.wer_metric(predictions, transcription_text).item() * 100

                total_loss += loss.item()
                total_wer += wer

                t.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss / (i + 1):.4f}',
                    'WER': f'{total_wer / (i + 1):.2f}%',
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Validation Loss'].append(total_loss / (i + 1))
                    self.step_history['Validation WER'].append(total_wer / (i + 1))
                    self.update_plot()

        val_loss = total_loss / len(self.test_loader)
        val_wer = total_wer / len(self.test_loader)

        self.history['Validation Loss'].append(val_loss)
        self.history['Validation WER'].append(val_wer)

        if self.scheduler:
            self.scheduler.step(val_wer)

        # Save best model based on WER (lower is better)
        if val_wer < self.best_metric:
            self.best_metric = val_wer
            self.save_checkpoint(is_best=True)

        self.logger.info(f"Validation loss for epoch {self.current_epoch}: {val_loss:.4f}")
        self.logger.info(f"Validation WER for epoch {self.current_epoch}: {val_wer:.2f}%")
        self.logger.info(f"Best WER so far: {self.best_metric:.2f}%\n")
        
        if self.scheduler:
            self.logger.info(f"Current Learning rate: {self.scheduler.get_last_lr()}")

        return
    

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run full training loop."""
        
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {self.current_epoch}")
            self.logger.log_training_resume(
                epoch=self.current_epoch, 
                global_step=self.current_step, 
                total_epochs=self.num_epochs
            )
        else:
            self.logger.info(f"Starting training for {self.num_epochs} epochs from scratch")
    
        print(f"Starting training from epoch {self.current_epoch} to {self.num_epochs}")

        for epoch in range(self.current_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            self.train_epoch()
            
            if self.test_loader:
                self.test_epoch()
    
            self.save_checkpoint()
        
        self.logger.info(f"Training completed! Best WER: {self.best_metric:.2f}%")
        
        return
    

    def save_checkpoint(self, is_best: Optional[bool] = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_history': self.step_history,
            'history': self.history,
            'best_metric': self.best_metric
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(self.checkpoint_path, 'best_model.pth')
        else:
            path = os.path.join(
                self.checkpoint_path, 
                f'model_epoch_{self.current_epoch}.pth'
            )

        torch.save(checkpoint, path)
        
        if self.verbose:
            save_type = "Best model" if is_best else "Checkpoint"
            self.logger.info(f"{save_type} saved to {path}")


    def load_checkpoint(self, checkpoint: Optional[str] = None, resume_from_best: Optional[bool] = False):
        """Load model checkpoint."""
        
        if resume_from_best:
            checkpoint_path = os.path.join(self.checkpoint_path, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch') + 1
        self.current_step = checkpoint.get('current_step')
        self.best_metric = checkpoint.get('best_metric')
        
        loaded_history = checkpoint.get('history')
        for key in self.history:
            self.history[key] = loaded_history.get(key, self.history[key])

        loaded_step_history = checkpoint.get('step_history')
        for key in self.step_history:
            self.step_history[key] = loaded_step_history.get(key, self.step_history[key])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return
