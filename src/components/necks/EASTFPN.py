import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kersize, stride, padding) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kersize, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kersize, stride, padding) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kersize, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return x

class EASTFPN(nn.Module):
    """
    @Params:
    'in_channels': channels from backbone's output
    """
    def __init__(self, backbone_out_channels, neck_out_channels) -> None:
        super().__init__()
        
        self.stem_out_conv = ConvBlock(backbone_out_channels, 16, kersize=7, stride=1, padding=3)
        
        self.conv_stage_1 = ConvBlock(16, out_channels=64, kersize=3, stride=2, padding=1)
        self.conv_stage_2 = ConvBlock(in_channels=64, out_channels=128, kersize=3, stride=2, padding=1)
        self.conv_stage_3 = ConvBlock(in_channels=128, out_channels=256, kersize=3, stride=2, padding=1)
        self.conv_stage_4 = ConvBlock(in_channels=256, out_channels=384, kersize=3, stride=2, padding=1)
        
        self.stage_g1 = DeConvBlock(384, neck_out_channels, kersize=4, stride=2, padding=1)
        self.stage_g2 = DeConvBlock(neck_out_channels, neck_out_channels, kersize=4, stride=2, padding=1) # Stage_3 + neck_out_channels
        self.stage_g3 = DeConvBlock(neck_out_channels, neck_out_channels, kersize=4, stride=2, padding=1)
        self.stage_g4 = ConvBlock(neck_out_channels, neck_out_channels, kersize=3, stride=1, padding=0)
        
        self.merging_conv_h2 = nn.Sequential(
            ConvBlock(in_channels=256 + neck_out_channels, out_channels=neck_out_channels, kersize=3, stride=1, padding=1),
            ConvBlock(in_channels=neck_out_channels, out_channels=neck_out_channels, kersize=1, stride=1, padding=0)
        )
        self.merging_conv_h3 = nn.Sequential(
            ConvBlock(in_channels=neck_out_channels + 128, out_channels=neck_out_channels, kersize=3, stride=1, padding=1),
            ConvBlock(in_channels=neck_out_channels, out_channels=neck_out_channels, kersize=1, stride=1, padding=0)
        )
        self.merging_conv_h4 = nn.Sequential(
            ConvBlock(in_channels=neck_out_channels + 64, out_channels=neck_out_channels, kersize=3, stride=1, padding=1),
            ConvBlock(in_channels=neck_out_channels, out_channels=neck_out_channels, kersize=1, stride=1, padding=0)
        )
        
        self.out_conv = ConvBlock(in_channels=neck_out_channels, out_channels=neck_out_channels, kersize=3, stride=1, padding=0)

    def forward(self, x):
        x = self.stem_out_conv(x)
        x = self.conv_stage_1(x)
        f4 = x
        x = self.conv_stage_2(x)
        f3 = x
        x = self.conv_stage_3(x)
        f2 = x
        x = self.conv_stage_4(x)        
        
        g1 = self.stage_g1(x) # f1
        h1 = torch.cat([f2, g1], dim=1)
        h1 = self.merging_conv_h2(h1)
        g2 = self.stage_g2(h1)
        h2 = torch.cat([f3, g2], dim=1)
        h2 = self.merging_conv_h3(h2)
        g3 = self.stage_g3(h2)
        h3 = torch.cat([f4, g3], dim=1)
        h3 = self.merging_conv_h4(h3)
        
        return h3