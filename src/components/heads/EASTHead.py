import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kersize, stride, padding, if_activation:bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kersize, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.isActivation = if_activation
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.isActivation: 
            x = self.relu(x)
        return x

class EASTHead(nn.Module):
    def __init__(self, neck_out_channels:int = 64):
        super().__init__()
        self.head_conv = ConvBlock(neck_out_channels, neck_out_channels, 3, 1, 1)
        self.score_conv = ConvBlock(neck_out_channels, 1, 1, 1, 0, False)
        self.geo_conv = ConvBlock(neck_out_channels , 8, 1, 1, 0, False)
        
    def forward(self, x):
        # X is output from neck
        x = self.head_conv(x)
        y_score = self.score_conv(x)
        y_geo = (self.geo_conv(x) - 0.5) * 2 + 800 # 1000 here's the hand-pick value for easier loss backward. 
        
        return {"y_score":y_score, "y_geo":y_geo}
    
        