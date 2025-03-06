import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class LossComputer:
   
    def __init__(self, config):
        self.config = config
        
    
        self.lambda1 = config.lambda1  # structure loss weight
        self.lambda2 = config.lambda2  # texture loss weight
        self.lambda3 = config.lambda3  # gap loss weight
        self.lambda4 = config.lambda4  # category loss weight
        
     
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_structure_loss(self, source_features: torch.Tensor, 
                             target_features: torch.Tensor) -> torch.Tensor:
       
        N = source_features.size(0)
        loss = (1.0 / N) * torch.sum((source_features - target_features) ** 2)
        return loss

    def compute_texture_loss(self, source_features: torch.Tensor, 
                           target_features: torch.Tensor) -> torch.Tensor:
       
        source_norm = F.normalize(source_features, p=2, dim=1)
        target_norm = F.normalize(target_features, p=2, dim=1)
        
       
        N = source_features.size(0)
        similarity = torch.sum(source_norm * target_norm, dim=1)
        loss = -(1.0 / N) * torch.sum(similarity)
        
        return loss

    def compute_gap_loss(self, source_total: torch.Tensor, 
                        target_structure: torch.Tensor) -> torch.Tensor:
       
        loss = torch.mean((source_total - target_structure) ** 2)
        return loss

    def compute_category_loss(self, logits: torch.Tensor, 
                            labels: torch.Tensor) -> torch.Tensor:
       
        return self.ce_loss(logits, labels)

    def compute_total_loss(self, 
                          source_outputs: Dict[str, torch.Tensor],
                          target_outputs: Dict[str, torch.Tensor],
                          labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
      
        structure_loss = self.compute_structure_loss(
            source_outputs['structure_features'],
            target_outputs['structure_features']
        )
        
     
        texture_loss = self.compute_texture_loss(
            source_outputs['texture_features'],
            target_outputs['texture_features']
        )
        
       
        gap_loss = self.compute_gap_loss(
            source_outputs['features'],
            target_outputs['structure_features']
        )
        
      
        category_loss = self.compute_category_loss(
            source_outputs['logits'],
            labels
        )
        
       
        total_loss = (self.lambda1 * structure_loss +
                     self.lambda2 * texture_loss +
                     self.lambda3 * gap_loss +
                     self.lambda4 * category_loss)
        
       
        loss_dict = {
            'total_loss': total_loss,
            'structure_loss': structure_loss,
            'texture_loss': texture_loss,
            'gap_loss': gap_loss,
            'category_loss': category_loss
        }
        
        return total_loss, loss_dict

def build_loss_computer(config) -> LossComputer:
   
    return LossComputer(config)
