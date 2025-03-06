import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Optional
import torchvision.transforms as transforms

class OracleDataset(Dataset):
 
    def __init__(self, 
                 root_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 is_source: bool = True):
    
        self.root_dir = root_dir
        self.transform = transform
        self.is_source = is_source
        self.image_paths = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
     
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(int(class_name))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
