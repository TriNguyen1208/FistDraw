import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn.functional as F
from src.draw_model.config.constant import NUM_EPOCHS, DEVICE, BATCH_SIZE, CHECKPOINT_PATH, TRAIN_SPLIT, CLASS_NAMES, THRESHOLD_PROB, DEVICE
from sklearn.metrics import accuracy_score
import copy
from src.draw_model.utils import save_model

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

    def fit(
        self,
        dataset: Dataset,
        loss: float,
        optimizer: torch.optim.Adam,
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(),
        start_epoch: int = 1
    ) -> None:
        if start_epoch > NUM_EPOCHS:
            print(f'Training complete with full epoch')
            return

        train_size = int(TRAIN_SPLIT * len(dataset))
        valid_size = len(dataset) - int(TRAIN_SPLIT * len(dataset))
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

        best_model = copy.deepcopy(self.state_dict())
        best_valid_loss = loss

        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            self.train()
            print(f'Running with epoch: {epoch} / {NUM_EPOCHS}')
            epoch_train_loss = 0
            total_loop_train = 0
            acc_score = 0

            #Train loop
            for batch in train_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                # train step
                logits = self(x_batch)

                #Calculate loss with prediction and true label with CrossEntropyLoss
                loss: torch.Tensor = criterion(logits, y_batch)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Sum of total loss in one epoch
                epoch_train_loss += loss.item()

                #Get the highest probability of each class
                pred = logits.argmax(dim = 1)
                F.softmax
                #Calculate the accuracy of prediction
                acc_score += accuracy_score(y_batch.cpu(), pred.cpu())
                total_loop_train += 1

            #Valid
            self.eval()
            with torch.no_grad():
                total_loop_valid = 0
                valid_score = 0
                epoch_valid_loss = 0
                for batch in valid_loader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                    logits = self(x_batch)
                    loss: torch.Tensor = criterion(logits, y_batch)
                    epoch_valid_loss += loss.item()
                    pred = logits.argmax(dim = 1)
                    valid_score += accuracy_score(y_batch.cpu(), pred.cpu())
                    total_loop_valid += 1

            acc_train_score = acc_score / total_loop_train
            acc_valid_score = valid_score / total_loop_valid
            loss_train = epoch_train_loss / len(train_loader)
            loss_valid = epoch_valid_loss / len(valid_loader)
            print(f'Training loss: {loss_train:.4f}. Training accuracy: {acc_train_score:.4f}')
            print(f'Validation loss: {loss_valid:.4f}. Validation accuracy: {acc_valid_score:.4f}')
            if loss_valid < best_valid_loss:
                best_valid_loss = loss_valid
                best_model = copy.deepcopy(self.state_dict())
                save_model(model=self, loss=best_valid_loss, optimizer=optimizer, epoch=epoch, file_name=CHECKPOINT_PATH)
                print(f'Best validation loss: {best_valid_loss:.4f}')
            self.load_state_dict(best_model)
            print(f'Model saved to {CHECKPOINT_PATH} successfully')

    def load(
        self,
        file_name: str
    ) -> nn.Module:
        checkpoint = torch.load(file_name, map_location=torch.device(DEVICE))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        return self

    def predict(self, input):
        self.load(CHECKPOINT_PATH).to(DEVICE)
        input = input.to(DEVICE)
        with torch.no_grad():
            logits = self(input)
            preds = F.softmax(logits, dim=1)
            values, indices = preds.max(dim=1)
            return [CLASS_NAMES[idx.item()] if value.item() >= THRESHOLD_PROB else "UNKNOWN" for value, idx in zip(values, indices)]
