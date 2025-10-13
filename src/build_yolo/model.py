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

# Loss function
class YOLOv1Loss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, targets):
        # preds, targets: [B, 7, 7, 5] -> (p, x, y, w, h)
        obj_mask = targets[..., 0] > 0  # cell có object
        noobj_mask = ~obj_mask

        # --- Cell có object ---
        pred_obj = preds[obj_mask]
        target_obj = targets[obj_mask]

        loss_coord = (
            self.mse(pred_obj[..., 1], target_obj[..., 1]) +
            self.mse(pred_obj[..., 2], target_obj[..., 2]) +
            self.mse(torch.sqrt(torch.clamp(pred_obj[..., 3], 1e-6, 1.0)),
                     torch.sqrt(target_obj[..., 3])) +
            self.mse(torch.sqrt(torch.clamp(pred_obj[..., 4], 1e-6, 1.0)),
                     torch.sqrt(target_obj[..., 4]))
        )

        loss_obj = self.mse(pred_obj[..., 0], target_obj[..., 0])
        loss_noobj = self.mse(preds[noobj_mask][..., 0], targets[noobj_mask][..., 0])

        total_loss = (
            self.lambda_coord * loss_coord
            + self.lambda_obj * loss_obj
            + self.lambda_noobj * loss_noobj
        )
        return total_loss


#training
device = "cuda"

# Dataset
train_loader = dt.train_loader2
valid_loader = dt.valid_loader2

# Model


# Training loop
if __name__ == "__main__":
    model = YOLOv1Tiny().to(device)
    criterion = YOLOv1Loss().to(device)
    try:
        model.load_state_dict(torch.load("src/build_yolo/tinyyolo_from_scratch.pth"))
        print("Loaded model")
    except FileNotFoundError:
        print("No saved model found, training from scratch!")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 15
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