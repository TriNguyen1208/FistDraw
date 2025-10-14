import torch
import torch.nn as nn
import src.build_yolo.dataset as dt

import torch
import torch.nn as nn

class YOLOv1Tiny(nn.Module):
    def __init__(self, S=7):
        super().__init__()
        self.S = S

        # --- Backbone CNN ---
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # 224→112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),                           # 112→56

            nn.Conv2d(64, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),                           # 56→28

            nn.Conv2d(192, 128, 1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),                           # 28→14

            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),                           # 14→7
        )

        # --- Prediction head ---
        self.pred = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * 5),  # (p, x, y, w, h)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)
        x = x.view(-1, self.S, self.S, 5)

        # tạo các phần tử mới (không ghi đè)
        p  = torch.sigmoid(x[..., 0:1])      # shape [...,1]
        xy = torch.sigmoid(x[..., 1:3])      # shape [...,2]
        wh = torch.clamp(x[..., 3:5], 0, 1)  # shape [...,2]

        out = torch.cat([p, xy, wh], dim=-1) # shape [...,5]
        return out
#--------------
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFistDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        in_features = self.backbone.fc.in_features  #int
        self.backbone.fc = nn.Identity()            #last fc <-- NULL
        self.box_head = nn.Linear(in_features, 4)  
        self.p_head = nn.Linear(in_features, 1) 

    def forward(self, x):
        # 1. Lấy feature vector từ backbone
        features = self.backbone(x)

        # 2. Chạy qua head để ra 5 giá trị cuối
        p = self.p_head(features)
        box = self.box_head(features)

        out = torch.cat([p, box], dim=1)  # (batch, 5) -> [p,x,y,w,h]
        return out


# --- Example usage ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetFistDetector(pretrained=True).to(device)

# Loss function
class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=5.0):
        super().__init__()
        self.cls_loss = nn.BCEWithLogitsLoss()       # cho p
        self.rgs_loss = nn.MSELoss()
        self.lambda_box = lambda_box

    def forward(self, outputs, targets):
        p_loss = self.cls_loss(outputs[:, 0], targets[:, 0])
        x_loss = self.rgs_loss(outputs[:, 1], targets[:, 1])
        y_loss = self.rgs_loss(outputs[:, 2], targets[:, 2])

        w = torch.sqrt(torch.clamp(outputs[:,3], min=1e-6))
        tw = torch.sqrt(torch.clamp(targets[:,3], min=1e-6))
        w_loss = self.rgs_loss(w, tw)

        h = torch.sqrt(torch.clamp(outputs[:,4], min=1e-6))
        th = torch.sqrt(torch.clamp(targets[:,4], min=1e-6))
        h_loss = self.rgs_loss(w, tw)

        box_loss = x_loss + y_loss + w_loss + h_loss
        total_loss = p_loss + self.lambda_box * box_loss

        # print(p_loss.tolist(), " - ", box_loss.tolist(), " | ", total_loss.tolist()) 
        return total_loss

#training
device = "cuda"

# Dataset
train_loader = dt.train_loader2
valid_loader = dt.valid_loader2

# Model


# Training loop
if __name__ == "__main__":
    model = ResNetFistDetector().to(device)
    criterion = DetectionLoss().to(device)
    try:
        model.load_state_dict(torch.load("src/build_yolo/tinyyolo_from_scratch.pth"))
        print("Loaded model")
    except FileNotFoundError:
        print("No saved model found, training from scratch!")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        #Validation
        model.eval() #switch to evaluation mode (no dropout, no dynamic weights)
        valid_loss = 0.0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                valid_loss += criterion(outputs, labels).item()
        print(f"[{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} - {valid_loss/len(valid_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "src/build_yolo/tinyyolo_from_scratch.pth")