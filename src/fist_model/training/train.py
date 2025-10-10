import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import src.fist_model.training.dataset as dt
import src.fist_model.model.model as md

# ----- Loss function custom -----
# class DetectionLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cls_loss = nn.BCEWithLogitsLoss()       # cho p
#         self.reg_loss = nn.SmoothL1Loss()  # cho w, h

#     def forward(self, outputs, targets):
#         p_loss = self.cls_loss(outputs[:, 0], targets[:, 0])
#         xy_loss = self.reg_loss(torch.sigmoid(outputs[:, 1:3]), targets[:, 1:3])
#         wh_loss = self.reg_loss(outputs[:, 3:5], targets[:, 3:5])
#         return p_loss + 5.0 * (xy_loss + wh_loss)

# class DetectionLoss(nn.Module):
#     def __init__(self, lambda_bbox=5.0):
#         super().__init__()
#         self.cls_loss = nn.BCEWithLogitsLoss()   # p: objectness
#         self.reg_loss = nn.SmoothL1Loss()        # bbox regression
#         self.lambda_bbox = lambda_bbox

#     def forward(self, outputs, targets):
#         """
#         outputs: (batch, 5) -> (p, x, y, w, h)
#         targets: (batch, 5) -> (p, x, y, w, h), x,y,w,h normalized [0,1]
#         """
#         # Objectness loss
#         p_loss = self.cls_loss(outputs[:, 0], targets[:, 0])

#         # Bbox center loss (linear, no sigmoid)
#         xy_loss = self.reg_loss(outputs[:, 1:3], targets[:, 1:3])

#         # Bbox size loss (use sqrt to reduce bias for large boxes)
#         w_pred_sqrt = torch.sqrt(outputs[:, 3].clamp(min=1e-6))
#         h_pred_sqrt = torch.sqrt(outputs[:, 4].clamp(min=1e-6))
#         w_target_sqrt = torch.sqrt(targets[:, 3])
#         h_target_sqrt = torch.sqrt(targets[:, 4])
#         wh_loss = self.reg_loss(w_pred_sqrt, w_target_sqrt) + self.reg_loss(h_pred_sqrt, h_target_sqrt)

#         # Total loss
#         total_loss = p_loss + self.lambda_bbox * (xy_loss + wh_loss)
#         return total_loss

class DetectionLoss(nn.Module):
    def __init__(self, lambda_bbox=5.0):
        super().__init__()
        self.bce = nn.BCELoss()       # objectness loss
        self.l1 = nn.SmoothL1Loss()   # bbox regression loss
        self.lambda_bbox = lambda_bbox

    def forward(self, outputs, targets):
        """
        outputs: (B, 5) -> (p, x, y, w, h)
        targets: (B, 5) -> (p, x, y, w, h), normalized [0,1]
        """

        # --- Objectness ---
        p_pred = torch.sigmoid(outputs[:, 0])
        p_true = targets[:, 0]
        loss_obj = self.bce(p_pred, p_true)

        # --- Bounding box (normalized prediction) ---
        bbox_pred = torch.sigmoid(outputs[:, 1:5])  # để giới hạn trong [0,1]
        bbox_true = targets[:, 1:5]
        loss_bbox = self.l1(bbox_pred, bbox_true)

        # --- Total loss ---
        total_loss = loss_obj + self.lambda_bbox * loss_bbox
        return total_loss

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