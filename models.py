from torchvision.models import swin_v2_b
from torchvision.models import Swin_V2_B_Weights
from torchvision.models import efficientnet_v2_s
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
from torchvision.models import regnet_y_16gf
from torchvision.models.regnet import RegNet_Y_16GF_Weights
from torch import nn
from torchvision import models


class EModel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.pretrained = pretrained
        self.FC = nn.Linear(1000, 4)

    def forward(self, x):
        x = self.pretrained(x)
        x = self.FC(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10000, 8000)
        self.fc2 = nn.Linear(8000, 5000)
        self.fc3 = nn.Linear(5000, 2000)
        self.fc4 = nn.Linear(2000, 1000)
        self.fc5 = nn.Linear(1000, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


resnet18 = EModel(models.resnet18(pretrained=True))
regnet_16gf = EModel(regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_V2))
effnetv2_s = EModel(efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1))
mlp = MLP()

