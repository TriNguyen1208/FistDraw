import torch
import torch.nn as nn
from math import pow

class FistDetection(nn.Module):
    def __init__(self, input_size = 224):
        super(FistDetection, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 5), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        # dimension = int(256 * pow(input_size / 4 - 3.5, 2))
        # dimension = (256 * 52 * 52)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8,8))
        dimension = 256*8*8
        self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, 5))        

    def forward(self, input: torch.tensor):
        '''
            input: torch.tensor (shape: (batch_size, channels, h, w)) 
            output: torch.tensor (shape: (batch_size, 5)) #(p, x, y, w, h)
        '''
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.adaptive_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        output[:, 0] = torch.sigmoid(output[:, 0]) #sigmoid for p
        # output = torch.sigmoid(output)
        return output