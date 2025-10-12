import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import src.fist_model.training.dataset as dt
import src.fist_model.model.model as md

# ----- Loss function custom -----
def iou(box1, box2):
    x1_min = box1[:,0] - box1[:,2]/2
    y1_min = box1[:,1] - box1[:,3]/2
    x1_max = box1[:,0] + box1[:,2]/2
    y1_max = box1[:,1] + box1[:,3]/2

    x2_min = box2[:,0] - box2[:,2]/2
    y2_min = box2[:,1] - box2[:,3]/2
    x2_max = box2[:,0] + box2[:,2]/2
    y2_max = box2[:,1] + box2[:,3]/2

    inter_w = torch.clamp(torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min), min = 0)
    inter_h = torch.clamp(torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min), min = 0)

    inter = inter_w * inter_h
    union = (box1[:,2]*box1[:,3] + box2[:,2]*box2[:,3] - inter + 1e-6)

    return inter/union

class DetectionLoss(nn.Module):
    def __init__(self, lambda_iou = 5.0):
        super().__init__()
        self.cls_loss = nn.BCEWithLogitsLoss()       # cho p
        self.lambda_iou = lambda_iou

    def forward(self, outputs, targets):
        p_loss = self.cls_loss(outputs[:, 0], targets[:, 0])
        iou_loss = 1 - iou(outputs[:, 1:5], targets[:, 1:5]).mean()
        return p_loss + self.lambda_iou * iou_loss

# ----- Training loop -----
def train(model, train_loader = dt.train_loader2, valid_loader = dt.valid_loader2, num_epochs=25, lr=1e-4, device="cuda"):
    model = model.to(device)
    criterion = DetectionLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)              # forward
            loss = criterion(outputs, labels)  # tính loss

            optimizer.zero_grad()
            loss.backward()                    # backward
            optimizer.step()                   # update weight

            running_loss += loss.item()

        #calc error on validation dataset
        model.eval() #switch to evaluation mode (no dropout, no dynamic weights)
        valid_loss = 0.0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                valid_loss += criterion(outputs, labels).item()
        print(f"[{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} - {valid_loss/len(valid_loader):.4f}")
    
    #Save model
    save_path = "src/fist_model/trained.pth"
    torch.save(model.state_dict(), save_path)
    print("Saved model!")    

if __name__ == "__main__":
    model = md.FistDetection()
    try:
        model.load_state_dict(torch.load("src/fist_model/trained.pth"))
        print("Loaded model")
    except FileNotFoundError:
        print("No saved model found, training from scratch!")

    train(model)
# print(torch.cuda.is_available()) 