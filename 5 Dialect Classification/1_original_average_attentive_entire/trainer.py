import os
import torch 
from torch import nn
import matplotlib.pyplot as plt
from typing import Optional
from logger import TrainingLogger
from tqdm import tqdm
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score


class DialectClassificationTrainer:
    """
    Trainer for Tamil Dialect Classification using Wav2Vec2.
    Tracks Loss, Macro-Precision, Macro-Recall, and Macro-F1.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader, 
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 loss_fn: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 num_classes: int = 4,
                 log_path: Optional[str] = './logs/training.log',
                 num_epochs: Optional[int] = 16,
                 checkpoint_path: Optional[str] = './checkpoints',
                 graph_path: Optional[str] = './graphs/model_metrics.png',
                 verbose: Optional[bool] = True,
                 device: Optional[torch.device] = None) -> None:
        
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

        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
        self.scheduler = scheduler

        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.graph_path = graph_path
        self.verbose = verbose
        self.loss_update_step = 15

        self.current_epoch = 1
        self.current_step = 1
        self.best_metric = 0.0  # Best F1 score (higher is better)

        # Metrics for classification (macro-averaged)
        self.num_classes = num_classes
        self.precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(self.device)
        self.recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(self.device)
        self.f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(self.device)

        self.history = {
            'Training Loss': [],
            'Training Precision': [],
            'Training Recall': [],
            'Training F1': [],
            'Testing Loss': [],
            'Testing Precision': [],
            'Testing Recall': [],
            'Testing F1': []
        }

        self.step_history = {
            'Training Loss': [],
            'Training Precision': [],
            'Training Recall': [],
            'Training F1': [],
            'Testing Loss': [],
            'Testing Precision': [],
            'Testing Recall': [],
            'Testing F1': []
        }


    def update_plot(self) -> None:
        """Generate 2x4 plot of training and testing metrics."""
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))

        # Row 1: Training metrics
        axs[0, 0].plot(self.step_history['Training Loss'], color='blue', label='Training Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        axs[0, 1].plot(self.step_history['Training Precision'], color='green', label='Training Precision')
        axs[0, 1].set_title('Training Precision (Macro)')
        axs[0, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 1].set_ylabel('Precision')
        axs[0, 1].legend()

        axs[0, 2].plot(self.step_history['Training Recall'], color='orange', label='Training Recall')
        axs[0, 2].set_title('Training Recall (Macro)')
        axs[0, 2].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 2].set_ylabel('Recall')
        axs[0, 2].legend()

        axs[0, 3].plot(self.step_history['Training F1'], color='red', label='Training F1')
        axs[0, 3].set_title('Training F1 Score (Macro)')
        axs[0, 3].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 3].set_ylabel('F1 Score')
        axs[0, 3].legend()

        # Row 2: Testing metrics
        axs[1, 0].plot(self.step_history['Testing Loss'], color='blue', label='Testing Loss')
        axs[1, 0].set_title('Testing Loss')
        axs[1, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()

        axs[1, 1].plot(self.step_history['Testing Precision'], color='green', label='Testing Precision')
        axs[1, 1].set_title('Testing Precision (Macro)')
        axs[1, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 1].set_ylabel('Precision')
        axs[1, 1].legend()

        axs[1, 2].plot(self.step_history['Testing Recall'], color='orange', label='Testing Recall')
        axs[1, 2].set_title('Testing Recall (Macro)')
        axs[1, 2].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 2].set_ylabel('Recall')
        axs[1, 2].legend()

        axs[1, 3].plot(self.step_history['Testing F1'], color='red', label='Testing F1')
        axs[1, 3].set_title('Testing F1 Score (Macro)')
        axs[1, 3].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 3].set_ylabel('F1 Score')
        axs[1, 3].legend()

        plt.tight_layout()
        plt.savefig(self.graph_path)
        plt.close(fig)

        return


    def train_epoch(self) -> None:

        self.model.train()
        total_loss = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        # Reset metrics
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Training)') as t:
            
            for i, batch in t:
                
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(input_values=input_values, attention_mask=attention_mask)
                loss = self.loss_fn(logits, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Compute predictions
                predictions = torch.argmax(logits, dim=-1)

                # Update metrics
                precision = self.precision_metric(predictions, labels)
                recall = self.recall_metric(predictions, labels)
                f1 = self.f1_metric(predictions, labels)

                total_loss += loss.item()
                total_precision += precision.item()
                total_recall += recall.item()
                total_f1 += f1.item()

                self.current_step += 1

                t.set_postfix({
                    'Loss': loss.item(),
                    'Avg Loss': total_loss / (i + 1),
                    'Precision': total_precision / (i + 1),
                    'Recall': total_recall / (i + 1),
                    'F1': total_f1 / (i + 1)
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Training Loss'].append(total_loss / (i + 1))
                    self.step_history['Training Precision'].append(total_precision / (i + 1))
                    self.step_history['Training Recall'].append(total_recall / (i + 1))
                    self.step_history['Training F1'].append(total_f1 / (i + 1))
                    self.update_plot()

        train_loss = total_loss / len(self.train_loader)
        train_precision = total_precision / len(self.train_loader)
        train_recall = total_recall / len(self.train_loader)
        train_f1 = total_f1 / len(self.train_loader)

        self.history['Training Loss'].append(train_loss)
        self.history['Training Precision'].append(train_precision)
        self.history['Training Recall'].append(train_recall)
        self.history['Training F1'].append(train_f1)
        
        self.logger.info(f"Training loss for epoch {self.current_epoch}: {train_loss:.4f}")
        self.logger.info(f"Training precision for epoch {self.current_epoch}: {train_precision:.4f}")
        self.logger.info(f"Training recall for epoch {self.current_epoch}: {train_recall:.4f}")
        self.logger.info(f"Training F1 for epoch {self.current_epoch}: {train_f1:.4f}\n")

        return


    def test_epoch(self) -> None:

        self.model.eval()
        total_loss = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        # Reset metrics
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

        with tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Testing)') as t:
            
            for i, batch in t:
                
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.no_grad():
                    logits = self.model(input_values=input_values, attention_mask=attention_mask)
                    loss = self.loss_fn(logits, labels)
                    predictions = torch.argmax(logits, dim=-1)
                
                # Update metrics
                precision = self.precision_metric(predictions, labels)
                recall = self.recall_metric(predictions, labels)
                f1 = self.f1_metric(predictions, labels)

                total_loss += loss.item()
                total_precision += precision.item()
                total_recall += recall.item()
                total_f1 += f1.item()

                t.set_postfix({
                    'Loss': loss.item(),
                    'Avg Loss': total_loss / (i + 1),
                    'Precision': total_precision / (i + 1),
                    'Recall': total_recall / (i + 1),
                    'F1': total_f1 / (i + 1)
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Testing Loss'].append(total_loss / (i + 1))
                    self.step_history['Testing Precision'].append(total_precision / (i + 1))
                    self.step_history['Testing Recall'].append(total_recall / (i + 1))
                    self.step_history['Testing F1'].append(total_f1 / (i + 1))
                    self.update_plot()

        test_loss = total_loss / len(self.test_loader)
        test_precision = total_precision / len(self.test_loader)
        test_recall = total_recall / len(self.test_loader)
        test_f1 = total_f1 / len(self.test_loader)

        self.history['Testing Loss'].append(test_loss)
        self.history['Testing Precision'].append(test_precision)
        self.history['Testing Recall'].append(test_recall)
        self.history['Testing F1'].append(test_f1)

        if self.scheduler:
            self.scheduler.step(test_f1)

        # Save best model based on F1 score (higher is better)
        if test_f1 > self.best_metric:
            self.best_metric = test_f1
            self.save_checkpoint(is_best=True)

        self.logger.info(f"Testing loss for epoch {self.current_epoch}: {test_loss:.4f}")
        self.logger.info(f"Testing precision for epoch {self.current_epoch}: {test_precision:.4f}")
        self.logger.info(f"Testing recall for epoch {self.current_epoch}: {test_recall:.4f}")
        self.logger.info(f"Testing F1 for epoch {self.current_epoch}: {test_f1:.4f}\n")
        
        if self.scheduler:
            self.logger.info(f"Current Learning rate: {self.scheduler.get_last_lr()}")

        return
    

    def train(self, resume_from: Optional[str] = None) -> None:
        
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
        
        return
    

    def save_checkpoint(self, is_best: Optional[bool] = False):

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
