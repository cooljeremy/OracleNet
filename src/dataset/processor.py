import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Tuple
from .dataset import OracleDataset
from ..utils.config import Config

class DataProcessor:
 
    def __init__(self, config: Config):
        self.config = config
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
        
    def _get_train_transform(self) -> transforms.Compose:
     
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _get_test_transform(self) -> transforms.Compose:
      
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        
        source_train_dataset = OracleDataset(
            root_dir=os.path.join(self.config.data_root, 'source/train'),
            transform=self.train_transform,
            is_source=True
        )
        source_test_dataset = OracleDataset(
            root_dir=os.path.join(self.config.data_root, 'source/test'),
            transform=self.test_transform,
            is_source=True
        )
        
       
        target_train_dataset = OracleDataset(
            root_dir=os.path.join(self.config.data_root, 'target/train'),
            transform=self.train_transform,
            is_source=False
        )
        target_test_dataset = OracleDataset(
            root_dir=os.path.join(self.config.data_root, 'target/test'),
            transform=self.test_transform,
            is_source=False
        )
        
      
        source_train_loader = DataLoader(
            source_train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        source_test_loader = DataLoader(
            source_test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        target_train_loader = DataLoader(
            target_train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        target_test_loader = DataLoader(
            target_test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return source_train_loader, source_test_loader, target_train_loader, target_test_loader