from src.components.backbones.ResNet18 import ResNet18
from src.components.necks.EASTFPN import EASTFPN
from src.utils.common import count_parameters
from torchvision.models import vgg16
import torch



if __name__ == "__main__":
    input = torch.randn((1, 3, 512, 512))

    backbone = ResNet18()
    output = backbone(input)
    print("Backbone params:", count_parameters(backbone))
    
    neck = EASTFPN(output.shape[1], 64)
    output = neck(output)
    print("Neck params:", count_parameters(neck))
    print(f"==>> output.shape: {output.shape}")
    
