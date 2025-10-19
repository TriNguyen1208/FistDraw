from torch.utils.data import Dataset
import torch
from src.draw_model.config.constant import FILE_NAME
import numpy as np

class QuickDrawDataset(Dataset):
    def __init__(self, file_names):
        self.file_names = file_names
        self.data_list = [np.load(f, mmap_mode='r') for f in file_names]
        self.labels = list(range(len(file_names)))

    def __len__(self):
        return sum(len(d) for d in self.data_list)

    def __getitem__(self, idx):
        total = 0
        for i, data in enumerate(self.data_list):
            if idx < total + len(data):
                item = data[idx - total].reshape(28, 28)
                item = torch.tensor(item, dtype=torch.float32).unsqueeze(0) / 255.0
                return item, self.labels[i]
            total += len(data)

dataset = QuickDrawDataset(FILE_NAME)