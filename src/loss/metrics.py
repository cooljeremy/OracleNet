import torch
import torch.nn.functional as F

class MetricComputer:
  
    @staticmethod
    def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
      
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        return correct / total * 100.0

    @staticmethod
    def compute_domain_distance(source_features: torch.Tensor, 
                              target_features: torch.Tensor) -> float:
     
        distance = F.mse_loss(source_features, target_features)
        return distance.item()
        
    @staticmethod
    def compute_confusion_matrix(predictions: torch.Tensor, 
                               labels: torch.Tensor, 
                               num_classes: int) -> torch.Tensor:
      
        return torch.zeros(num_classes, num_classes).scatter_(
            1,
            labels.view(-1, 1),
            1.0,
            reduce='add'
        )