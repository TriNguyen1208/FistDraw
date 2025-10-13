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

# class FistDetection(nn.Module):
#     def __init__(self, input_size = 224):
#         super(FistDetection, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
#         self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
#         self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
#         self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.ReLU(inplace=True))

#         self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
#         dimension = 128*4*4
#         # self.fc1 = nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(inplace=True), nn.Dropout(0.3))
#         # self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3))
#         # self.fc3 = nn.Sequential(nn.Linear(64, 5)) 

#         self.p_head = nn.Sequential(
#             nn.Linear(dimension, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
#             nn.Linear(64, 16), nn.ReLU(inplace=True), nn.Dropout(0.3),  
#             nn.Linear(16, 1)
#         )       

#         self.box_head = nn.Sequential(
#             nn.Linear(dimension, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
#             nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(0.3),  
#             nn.Linear(32, 4)
#         )   

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

#         p = self.p_head(output)
#         box = self.box_head(output)
#         output = torch.cat((p, box), dim=1)
        
#         return output

#==========================================================================

class FistDetection(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )

        # Head dự đoán objectness — có thể flatten vì chỉ cần biết có/không
        self.obj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # Head dự đoán bbox — giữ spatial bằng conv
        self.box_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.shared(x)
        p = self.obj_head(feat)
        box = self.box_head(feat)
        return torch.cat([p, box], dim=1)

#------------------------------------------------------------------------------------------
import torchvision.models as models
class ResNetFistDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # 1️⃣ Backbone - ResNet34 pretrained trên ImageNet
        base = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  
        # giữ lại tất cả trừ lớp FC và AvgPool cuối

        # 2️⃣ Global pooling để nén spatial info
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3️⃣ Head 1 — phân loại object (có/không)
        self.obj_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # không sigmoid — dùng BCEWithLogitsLoss
        )

        # 4️⃣ Head 2 — dự đoán bounding box
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 4), nn.Sigmoid()  # (x, y, w, h) normalized [0,1]
        )

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.global_pool(feat)

        p = self.obj_head(pooled)
        box = self.box_head(pooled)
        return torch.cat([p, box], dim=1)