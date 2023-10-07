import torch.nn as nn
from ..backbones.ResNet18 import ResNet, BasicBlock
from ..backbones.AnotherResNet18 import ResNetNew, block
from ..necks.EASTFPN import EASTFPN
from ..heads.EASTHead import EASTHead
from ...utils.common import count_parameters
from torchvision.models import mobilenet_v3_small

class EASTNet(nn.Module):
    def __init__(self):
        super(EASTNet, self).__init__()
        self.backbone = ResNet(img_channels=3, num_layers=18, block=BasicBlock, network_scale_channels=128)
        
        print("backbone params:", count_parameters(self.backbone))
        self.neck = EASTFPN(128, 64)
        print("neck params:", count_parameters(self.neck))
        self.head = EASTHead(neck_out_channels=64)
        print("head params:", count_parameters(self.head))
        
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
        