import torch.nn as nn
from src.draw_model.config.constant import DEVICE
class DrawModel(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.flat = nn.Flatten() #256 gia tri 4 * 4 * 16
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=120, device=DEVICE),
            nn.ReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84, device=DEVICE),
            nn.ReLU()
        )
        self.out = nn.Linear(in_features=84, out_features=num_classes, device=DEVICE)
    def forward(self, input):
        input = self.conv1(input)
        input = self.pool1(input)
        input = self.conv2(input)
        input = self.pool2(input)
        input = self.flat(input)
        input = self.fc_1(input)
        input = self.fc_2(input)
        return self.out(input)
    