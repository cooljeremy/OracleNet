import torch
import torch.nn as nn
from typing import Tuple

class TextureStructureDecouplingModule(nn.Module):
   
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
       
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
     
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
     
        self.structure_attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.texture_attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       
        structure_features = self.structure_encoder(x)
        structure_attention = self.structure_attention(structure_features)
        structure_out = structure_features * structure_attention
        
       
        texture_features = self.texture_encoder(x)
        texture_attention = self.texture_attention(texture_features)
        texture_out = texture_features * texture_attention
        
        return structure_out, texture_out