import torch
import torch.nn as nn
from math import pow

# class FistDetection(nn.Module):
#     def __init__(self, input_size = 224):
#         super(FistDetection, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 5), nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.ReLU(inplace=True))
#         self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
#         # dimension = int(256 * pow(input_size / 4 - 3.5, 2))
#         # dimension = (256 * 52 * 52)

#         self.adaptive_pool = nn.AdaptiveAvgPool2d((8,8))
#         dimension = 256*8*8
#         self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.ReLU(inplace=True), nn.Dropout(0.5))
#         self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.5))
#         self.fc3 = nn.Sequential(nn.Linear(128, 5))        

#     def forward(self, input: torch.tensor):
#         '''
#             input: torch.tensor (shape: (batch_size, channels, h, w)) 
#             output: torch.tensor (shape: (batch_size, 5)) #(p, x, y, w, h)
#         '''
#         output = self.conv1(input)
#         output = self.conv2(output)
#         output = self.conv3(output)
#         output = self.conv4(output)
#         output = self.adaptive_pool(output)
#         output = output.view(output.size(0), -1)
#         output = self.fc1(output)
#         output = self.fc2(output)
#         output = self.fc3(output)
        
#         return output
    
#==================================================================================

class FistDetection(nn.Module):
    def __init__(self, input_size = 224):
        super(FistDetection, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.ReLU(inplace=True))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
        dimension = 128*4*4
        self.fc1 = nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.fc3 = nn.Sequential(nn.Linear(64, 5))        

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
        
        return output

#==========================================================================

# class FistDetection(nn.Module):
#     def __init__(self, input_size=224):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
#         )

#         # 2 nhánh tách biệt: 1 cho objectness, 1 cho bbox
#         self.head_conf = nn.Sequential(
#             nn.Conv2d(128, 1, 1),
#             nn.Sigmoid()
#         )
#         self.head_bbox = nn.Conv2d(128, 4, 1)  # x, y, w, h (dạng heatmap)

#         self.pool = nn.AdaptiveAvgPool2d(1)  # gom spatial lại sau khi tính xong

#     def forward(self, x):
#         f = self.backbone(x)
#         conf = self.pool(self.head_conf(f)).squeeze(-1).squeeze(-1)
#         bbox = self.pool(self.head_bbox(f)).squeeze(-1).squeeze(-1)
#         out = torch.cat([conf, bbox], dim=1)  # (B,5)
#         return out