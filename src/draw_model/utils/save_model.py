import torch
import torch.nn as nn

def save_model(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    epoch: int,
    file_name: str,
    loss: float
) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, file_name)