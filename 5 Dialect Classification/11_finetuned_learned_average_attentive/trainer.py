"""
Trainer for Dialect Classification with frame-level features.
"""

import os
import torch 
from torch import nn
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import logging
from logging.handlers import RotatingFileHandler


class TrainingLogger:
    def __init__(self, log_path: str = './logs/training.log', level: int = logging.INFO):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.logger = logging.getLogger('DialectClassifier')
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        file_handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)


class DialectClassificationTrainer:
    def __init__(self, model, train_loader, test_loader=None, loss_fn=None, optimizer=None,
                 scheduler=None, num_classes=4, log_path='./logs/training.log', num_epochs=100,
                 checkpoint_path='./checkpoints', graph_path='./graphs/metrics.png',
                 verbose=True, device=None):
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        
        self.logger = TrainingLogger(log_path=log_path)
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
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
        self.best_metric = 0.0

        self.precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(self.device)
        self.recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(self.device)
        self.f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(self.device)

        self.history = {k: [] for k in ['Training Loss', 'Training Precision', 'Training Recall', 'Training F1',
                                         'Testing Loss', 'Testing Precision', 'Testing Recall', 'Testing F1']}
        self.step_history = {k: [] for k in self.history.keys()}

    def update_plot(self):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        metrics = ['Loss', 'Precision', 'Recall', 'F1']
        colors = ['blue', 'green', 'orange', 'red']
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            axs[0, i].plot(self.step_history[f'Training {metric}'], color=color)
            axs[0, i].set_title(f'Training {metric}')
            axs[1, i].plot(self.step_history[f'Testing {metric}'], color=color)
            axs[1, i].set_title(f'Testing {metric}')
        plt.tight_layout()
        plt.savefig(self.graph_path)
        plt.close(fig)

    def train_epoch(self):
        self.model.train()
        total_loss, total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0
        self.precision_metric.reset(); self.recall_metric.reset(); self.f1_metric.reset()

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                  desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Train)') as t:
            for i, batch in t:
                embeddings = batch['embeddings'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(embeddings, lengths)
                loss = self.loss_fn(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                predictions = torch.argmax(logits, dim=-1)
                precision = self.precision_metric(predictions, labels)
                recall = self.recall_metric(predictions, labels)
                f1 = self.f1_metric(predictions, labels)

                total_loss += loss.item()
                total_precision += precision.item()
                total_recall += recall.item()
                total_f1 += f1.item()
                self.current_step += 1
                t.set_postfix({'Loss': f'{loss.item():.4f}', 'F1': f'{total_f1/(i+1):.4f}'})

                if i % self.loss_update_step == 0 and i != 0:
                    for k, v in [('Loss', total_loss), ('Precision', total_precision), 
                                 ('Recall', total_recall), ('F1', total_f1)]:
                        self.step_history[f'Training {k}'].append(v / (i + 1))
                    self.update_plot()

        n = len(self.train_loader)
        for k, v in [('Loss', total_loss), ('Precision', total_precision), 
                     ('Recall', total_recall), ('F1', total_f1)]:
            self.history[f'Training {k}'].append(v / n)
        self.logger.info(f"Epoch {self.current_epoch} Train - Loss: {total_loss/n:.4f}, F1: {total_f1/n:.4f}")

    def test_epoch(self):
        self.model.eval()
        total_loss, total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0
        self.precision_metric.reset(); self.recall_metric.reset(); self.f1_metric.reset()

        with tqdm(enumerate(self.test_loader), total=len(self.test_loader),
                  desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Test)') as t:
            for i, batch in t:
                embeddings = batch['embeddings'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.no_grad():
                    logits = self.model(embeddings, lengths)
                    loss = self.loss_fn(logits, labels)
                    predictions = torch.argmax(logits, dim=-1)
                
                precision = self.precision_metric(predictions, labels)
                recall = self.recall_metric(predictions, labels)
                f1 = self.f1_metric(predictions, labels)

                total_loss += loss.item()
                total_precision += precision.item()
                total_recall += recall.item()
                total_f1 += f1.item()
                t.set_postfix({'Loss': f'{loss.item():.4f}', 'F1': f'{total_f1/(i+1):.4f}'})

                if i % self.loss_update_step == 0 and i != 0:
                    for k, v in [('Loss', total_loss), ('Precision', total_precision), 
                                 ('Recall', total_recall), ('F1', total_f1)]:
                        self.step_history[f'Testing {k}'].append(v / (i + 1))
                    self.update_plot()

        n = len(self.test_loader)
        test_f1 = total_f1 / n
        for k, v in [('Loss', total_loss), ('Precision', total_precision), 
                     ('Recall', total_recall), ('F1', total_f1)]:
            self.history[f'Testing {k}'].append(v / n)

        if self.scheduler: self.scheduler.step(test_f1)
        if test_f1 > self.best_metric:
            self.best_metric = test_f1
            self.save_checkpoint(is_best=True)
        self.logger.info(f"Epoch {self.current_epoch} Test - Loss: {total_loss/n:.4f}, F1: {test_f1:.4f}")

    def train(self, resume_from=None):
        if resume_from: self.load_checkpoint(resume_from)
        else: self.logger.info(f"Starting training for {self.num_epochs} epochs")
        for epoch in range(self.current_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            self.train_epoch()
            if self.test_loader: self.test_epoch()
            self.save_checkpoint()

    def save_checkpoint(self, is_best=False):
        checkpoint = {'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(), 'history': self.history,
                      'step_history': self.step_history, 'best_metric': self.best_metric}
        if self.scheduler: checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        path = os.path.join(self.checkpoint_path, 'best_model.pth' if is_best else f'model_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint, path)
        if self.verbose: self.logger.info(f"{'Best' if is_best else 'Checkpoint'} saved to {path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.history = checkpoint.get('history', self.history)
        self.step_history = checkpoint.get('step_history', self.step_history)
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
