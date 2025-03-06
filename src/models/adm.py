import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AdaptiveDeformationModule(nn.Module):
   
    def __init__(self, config):
        super().__init__()
        self.num_control_points = config.num_control_points
        self.displacement = config.displacement
        self.alpha1 = config.alpha1
        self.alpha2 = config.alpha2
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
       
        self.control_points_predictor = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_control_points * 2, kernel_size=1)
        )

    def compute_edge_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        
     
        Ix = F.conv2d(x, sobel_x.repeat(x.size(1), 1, 1, 1), padding=1, groups=x.size(1))
        Iy = F.conv2d(x, sobel_y.repeat(x.size(1), 1, 1, 1), padding=1, groups=x.size(1))
        
        
        magnitude = torch.sqrt(Ix.pow(2) + Iy.pow(2))
        orientation = torch.atan2(Iy, Ix)
        
        return magnitude, orientation

    def compute_local_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        squared = x.pow(2)
        local_var = F.avg_pool2d(squared, kernel_size=3, stride=1, padding=1) - mean.pow(2)
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-10))
        
        
        local_contrast = (x - mean) / (local_std + 1e-6)
        
       
        global_std = torch.std(x, dim=(2, 3), keepdim=True)
        noise_suppression = 1 - (local_std / (global_std + 1e-6))
        
        return local_contrast, noise_suppression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
      
        edge_magnitude, edge_orientation = self.compute_edge_features(x)
        
       
        local_contrast, noise_suppression = self.compute_local_features(x)
        
        
        features = self.conv(x)
        
      
        control_points = self.control_points_predictor(features)
        control_points = control_points.view(batch_size, self.num_control_points, 2)
        
       
        theta = (self.alpha1 * edge_magnitude + self.alpha2 * torch.cos(edge_orientation) + 
                self.beta1 * local_contrast + self.beta2 * noise_suppression)
        
       
        grid = F.affine_grid(theta.view(-1, 2, 3), x.size())
        deformed = F.grid_sample(x, grid)
        
        return deformed
