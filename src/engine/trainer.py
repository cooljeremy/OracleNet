import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict

from ..loss.metrics import MetricComputer

def training_step(model: nn.Module,
                 loss_computer,
                 source_batch: torch.Tensor,
                 target_batch: torch.Tensor,
                 labels: torch.Tensor,
                 optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    
    source_outputs = model(source_batch, is_source=True)
    target_outputs = model(target_batch, is_source=False)
    

    total_loss, loss_dict = loss_computer.compute_total_loss(
        source_outputs, target_outputs, labels
    )
    

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
  
    accuracy = MetricComputer.compute_accuracy(
        source_outputs['logits'], labels
    )
    domain_distance = MetricComputer.compute_domain_distance(
        source_outputs['features'], target_outputs['features']
    )
    
   
    metrics = {
        'accuracy': accuracy,
        'domain_distance': domain_distance
    }
    metrics.update({k: v.item() for k, v in loss_dict.items()})
    
    return metrics

class Trainer:
 
    def __init__(self,
                 config,
                 model: nn.Module,
                 loss_computer,
                 source_train_loader: DataLoader,
                 target_train_loader: DataLoader,
                 source_val_loader: DataLoader,
                 target_val_loader: DataLoader):
        self.config = config
        self.model = model
        self.loss_computer = loss_computer
        self.source_train_loader = source_train_loader
        self.target_train_loader = target_train_loader
        self.source_val_loader = source_val_loader
        self.target_val_loader = target_val_loader
        
      
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
        
       
        self.save_dir = Path('checkpoints')
        self.save_dir.mkdir(exist_ok=True)
        
       
        self.best_accuracy = 0.0
        self.best_epoch = 0
        
      
        self.train_records = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': []
        }

    def save_checkpoint(self, epoch: int, accuracy: float):
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'best_accuracy': self.best_accuracy,
            'train_records': self.train_records
        }
        
       
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
       
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            logging.info(f'best accuracy: {accuracy:.2f}%')

    def load_checkpoint(self, checkpoint_path: str):
       
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint
