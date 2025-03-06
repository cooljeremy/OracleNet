import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple

from ..loss.metrics import MetricComputer

class Evaluator:
   
    def __init__(self, config, model: nn.Module):
        self.config = config
        self.model = model

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
     
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        for batch, labels in tqdm(test_loader, desc='Evaluating'):
           
            batch = batch.to(self.config.device)
            labels = labels.to(self.config.device)
            
          
            outputs = self.model(batch)
            predictions = torch.argmax(outputs['logits'], dim=1)
            
         
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    
        predictions = np.array(all_predictions)
        ground_truth = np.array(all_labels)
        
        
        accuracy = (predictions == ground_truth).mean() * 100
        
        return accuracy, predictions, ground_truth

    def analyze_results(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
        
        classes = np.unique(ground_truth)
        class_accuracies = {}
        
        for cls in classes:
            mask = ground_truth == cls
            class_acc = (predictions[mask] == ground_truth[mask]).mean() * 100
            class_accuracies[cls] = class_acc
        
      
        best_class = max(class_accuracies, key=class_accuracies.get)
        worst_class = min(class_accuracies, key=class_accuracies.get)
        
      
        confusion_matrix = MetricComputer.compute_confusion_matrix(
            torch.from_numpy(predictions),
            torch.from_numpy(ground_truth),
            self.config.num_classes
        )
        
        return {
            'class_accuracies': class_accuracies,
            'best_class': (best_class, class_accuracies[best_class]),
            'worst_class': (worst_class, class_accuracies[worst_class]),
            'confusion_matrix': confusion_matrix
        }

def build_evaluator(config, model: nn.Module) -> Evaluator:
   
    return Evaluator(config=config, model=model)