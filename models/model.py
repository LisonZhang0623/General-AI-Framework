import torch
import torch.nn as nn
import torch.nn.functional as F

class EasyModel(nn.Module):
    def __init__(self):
        super(EasyModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, 7, 3,1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1, 1),
        )
    def forward(self, x):
        H, W = x.shape[-2:]
        out = self.conv(x)
        out = F.interpolate(out, size=[H, W], mode="bilinear", align_corners=False)
        return out

if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 224, 224)
    model = EasyModel()
    print(model(input_tensor).size())