import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import torch
from sklearn.manifold import TSNE

class Visualizer:
   
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def plot_training_curves(self, 
                           train_records: Dict[str, List[float]], 
                           save_name: str = 'training_curves.png'):
     
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
       
        ax1.plot(train_records['train_losses'], label='Train Loss')
        ax1.plot(train_records['val_losses'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
    
        ax2.plot(train_records['train_accuracies'], label='Train Accuracy')
        ax2.plot(train_records['val_accuracies'], label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()

    def plot_confusion_matrix(self,
                            confusion_matrix: np.ndarray,
                            save_name: str = 'confusion_matrix.png'):
    
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.save_dir / save_name)
        plt.close()

    def visualize_features(self, 
                          source_features: np.ndarray,
                          target_features: np.ndarray,
                          labels: np.ndarray,
                          save_name: str = 'feature_distribution.png'):
       
        combined_features = np.vstack([source_features, target_features])
        
      
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(combined_features)
        
   
        n_source = source_features.shape[0]
        source_2d = features_2d[:n_source]
        target_2d = features_2d[n_source:]
        
      
        plt.figure(figsize=(10, 8))
        plt.scatter(source_2d[:, 0], source_2d[:, 1], c=labels[:n_source], 
                   cmap='tab10', marker='o', label='Source')
        plt.scatter(target_2d[:, 0], target_2d[:, 1], c=labels[n_source:],
                   cmap='tab10', marker='^', label='Target')
        plt.title('Feature Distribution (t-SNE)')
        plt.legend()
        plt.savefig(self.save_dir / save_name)
        plt.close()

    def visualize_attention(self,
                          image: torch.Tensor,
                          attention_map: torch.Tensor,
                          save_name: str = 'attention.png'):
       
        image = image.cpu().numpy().transpose(1, 2, 0)
        attention = attention_map.cpu().numpy()
        
       
        image = (image - image.min()) / (image.max() - image.min())
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        
       
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(attention, cmap='jet')
        ax2.set_title('Attention Map')
        ax2.axis('off')
        
       
        ax3.imshow(image)
        ax3.imshow(attention, cmap='jet', alpha=0.5)
        ax3.set_title('Overlayed')
        ax3.axis('off')
        
        plt.savefig(self.save_dir / save_name)
        plt.close()
