import torch
import torch.nn as nn
import src.build_yolo.dataset as dt

class TinyYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 5, 1)  # x, y, w, h, confidence
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.head(f)
        # Flatten to [batch, 5] assuming 1 object
        out = out.mean([2,3])  # global average pool
        return out

# Loss function
def yolo_loss(pred, target):
    # target = [x, y, w, h, obj_conf]
    bbox_loss = nn.MSELoss()(pred[:,:4], target[:,:4])
    conf_loss = nn.BCEWithLogitsLoss()(pred[:,4], target[:,4])
    return bbox_loss + conf_loss

#training
device = "cuda"

# Dataset
train_loader = dt.train_loader2
valid_loader = dt.valid_loader2

# Model
model = TinyYOLO().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
if __name__ == "__main__":
    epochs = 25
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        valid_loss = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = yolo_loss(preds, targets)
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
                valid_loss += yolo_loss(outputs, labels).item()
        print(f"[{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} - {valid_loss/len(valid_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "tinyyolo_from_scratch.pth")