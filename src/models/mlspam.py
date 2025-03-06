import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLevelAttentionModule(nn.Module):

    def __init__(self, in_channels: int = 64):
        super().__init__()
        
       
        self.micro_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
     
        self.macro_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
      
        self.micro_attention = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
       
        self.macro_attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
      
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, in_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        micro_features = self.micro_conv(x)
        micro_attention = self.micro_attention(micro_features)
        micro_out = micro_features * micro_attention
        
     
        macro_features = self.macro_conv(x)
        macro_attention = self.macro_attention(macro_features)
        macro_out = macro_features * macro_attention
        
      
        macro_out = F.interpolate(macro_out, size=micro_out.shape[2:], mode='bilinear', align_corners=False)
        
      
        combined = torch.cat([micro_out, macro_out], dim=1)
        out = self.fusion(combined)
        
        return out