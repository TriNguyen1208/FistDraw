import torch
import torch.nn as nn
from src.draw_model.config.constant import DEVICE
def load_model(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    file_name: str,
) -> tuple[int, int]:
    checkpoint = torch.load(file_name, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return loss, epoch + 1