import torch.nn as nn
import torch
from src.draw_model.model.model import DrawModel
from src.draw_model.config.constant import DEVICE, NUM_CLASSES

def load_model(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    file_name: str
) -> int:
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def save_model(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    epoch: int,
    file_name: str
) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_name)

def load_predict_model(
    file_name: str
) -> nn.Module:
    model = DrawModel(num_classes=NUM_CLASSES)
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model