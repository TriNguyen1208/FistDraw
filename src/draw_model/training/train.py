import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.draw_model.config.constant import NUM_EPOCHS, DEVICE, BATCH_SIZE, NUM_CLASSES, LEARNING_RATE, CHECKPOINT_PATH, TRAIN_SPLIT  
from src.draw_model.config.config import IMAGE_DATA, LABEL_DATA
from torch.nn import functional as F
from src.draw_model.model.model import DrawModel
import os
from src.draw_model.training.checkpoint import save_model, load_model  
from sklearn.metrics import accuracy_score
def train_model(
        model: nn.Module, 
        optimizer: torch.optim.Adam, 
        data: tuple[torch.Tensor, torch.Tensor], 
        criterion: nn.CrossEntropyLoss, 
        start_epoch: int = 1
) -> None:
    if start_epoch > NUM_EPOCHS:
        print(f'Training complete with full epoch')
        return
    
    images, labels = data
    dataset = TensorDataset(images, labels)
    train_size = int(TRAIN_SPLIT * len(dataset))
    valid_size = len(dataset) - int(TRAIN_SPLIT * len(dataset))
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        print(f'Running with epoch: {epoch} / {NUM_EPOCHS}')
        epoch_train_loss = 0
        total_loop_train = 0
        acc_score = 0

        #Train loop
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            # train step
            logits = model(x_batch)
            
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
            
            #Calculate the accuracy of prediction
            acc_score += accuracy_score(pred, y_batch)
            total_loop_train += 1
        
        #Valid 
        model.eval()
        with torch.no_grad():
            total_loop_valid = 0
            valid_score = 0
            epoch_valid_loss = 0
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(x_batch)
                loss: torch.Tensor = criterion(logits, y_batch)
                epoch_valid_loss += loss.item() 
                pred = logits.argmax(dim = 1)
                valid_score += accuracy_score(pred, y_batch)
                total_loop_valid += 1

        acc_train_score = acc_score / total_loop_train
        acc_valid_score = valid_score / total_loop_valid
        loss_train = epoch_train_loss / len(train_loader)
        loss_valid = epoch_valid_loss / len(valid_loader)
        print(f'Training loss: {loss_train:.4f}. Training accuracy: {acc_train_score:.4f}')
        print(f'Validation loss: {loss_valid:.4f}. Validation accuracy: {acc_valid_score:.4f}')
        save_model(model=model, optimizer=optimizer, epoch=epoch, file_name=CHECKPOINT_PATH)
        print(f'Model saved to {CHECKPOINT_PATH} successfully')

model = DrawModel(num_classes=NUM_CLASSES)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

start_epoch = 1
try:
    if os.path.exists(CHECKPOINT_PATH):
        start_epoch = load_model(model, optimizer, CHECKPOINT_PATH) + 1
        print(f'Resume training from epoch {start_epoch}')
except Exception as e:
    print(f'Error loading model: {e}')  

train_model(model=model, optimizer=optimizer, data=(IMAGE_DATA, LABEL_DATA), criterion=criterion, start_epoch=start_epoch)