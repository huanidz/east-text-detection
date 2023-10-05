import torch.nn as nn
from ..backbones.ResNet18 import ResNet18
from ..necks.EASTFPN import EASTFPN
from ..heads.EASTHead import EASTHead

class EASTNet(nn.Module):
    def __init__(self):
        super(EASTNet, self).__init__()
        self.backbone = ResNet18()
        self.neck = EASTFPN(64, 64)
        self.head = EASTHead(neck_out_channels=64)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
        