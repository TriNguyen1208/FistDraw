import torch.nn as nn
import torch
from src.draw_model.training.checkpoint import load_predict_model
from src.draw_model.config.constant import CHECKPOINT_PATH, DEVICE, CLASS_NAMES

def predict_model(
    model: nn.Module,
    input: torch.Tensor #(N, 1, 28, 28)
) -> str:
    input = input.to(DEVICE)
    with torch.no_grad():
        logits = model(input)
        pred = logits.argmax(dim = 1)
        return CLASS_NAMES[pred.item()]
    
model = load_predict_model(CHECKPOINT_PATH).to(DEVICE)
class_name = predict_model(model, torch.randn(1, 1, 28, 28))
print(class_name)