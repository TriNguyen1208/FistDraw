import torch
from src.draw_model.data.preprocessing import Preprocessing 
from src.draw_model.config.constant import FILE_NAME

IMAGE_DATA: torch.Tensor = []
LABEL_DATA: torch.Tensor = []
for i in range(len(FILE_NAME)):
    data = torch.from_numpy(Preprocessing.get_data(FILE_NAME[i]))
    x_data = data.reshape(-1, 28, 28)
    y_data = torch.full(size=(x_data.shape[0], ), fill_value=i)
    IMAGE_DATA.append(x_data)
    LABEL_DATA.append(y_data)

IMAGE_DATA: torch.Tensor = torch.cat(IMAGE_DATA, dim=0)
LABEL_DATA: torch.Tensor = torch.cat(LABEL_DATA, dim=0)
IMAGE_DATA: torch.Tensor = IMAGE_DATA / 255.0
IMAGE_DATA:torch.Tensor = IMAGE_DATA.unsqueeze(1)