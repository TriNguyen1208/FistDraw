import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import src.fist_model.training.dataset as dt
import src.fist_model.model.model as md

# ----- Loss function custom -----
class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.BCELoss()       # cho p
        self.reg_loss = nn.SmoothL1Loss()  # cho bbox

    def forward(self, outputs, targets):
        p_loss = self.cls_loss(outputs[:, 0], targets[:, 0])
        box_loss = self.reg_loss(outputs[:, 1:], targets[:, 1:])
        return p_loss + box_loss


# ----- Training loop -----
def train(model, train_loader = dt.train_loader, valid_loader = dt.valid_loader, num_epochs=5, lr=1e-3, device="cuda"):
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
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

        #calc error on validation dataset
        model.eval() #switch to evaluation mode (no dropout, no dynamic weights)
        valid_loss = 0.0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                valid_loss += criterion(outputs, labels).item()
        print(f"Valid [{epoch+1}/{num_epochs}] Loss: {valid_loss/len(valid_loader):.4f}")
    
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