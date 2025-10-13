import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from src.fist_model.config import constant as c

# # Dùng
# for imgs, labels in train_loader:
#     print(imgs.shape)   # (32, 3, H, W)
#     print(labels.shape) # (32, 5)
#     break

#---------------------------------------------------------------------------------------
import random
class FistDataset(Dataset):
    def __init__(self, img_folder, label_folder, augment=True):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.files = sorted(os.listdir(img_folder))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.files[idx])
        label_path = os.path.join(self.label_folder, self.files[idx].replace('.jpg', '.txt'))

        # --- Đọc ảnh ---
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # --- Tensor hóa ---
        cv2.resize(img, (224, 224))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [C, H, W], float32

        # --- Target tensor ---
        S = 7
        target_tensor = torch.zeros((S, S, 5), dtype=torch.float32)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                nums = [float(x) for x in f.read().split()]  # x y w h
            if len(nums) >= 4:
                x, y, w, h = nums[:4]

                # clamp để tránh tràn chỉ số
                x = min(max(x, 0.0), 0.999)
                y = min(max(y, 0.0), 0.999)

                # cell mà object nằm trong (YOLO: x=cột, y=hàng)
                cell_x = int(x * S)
                cell_y = int(y * S)

                # tọa độ tương đối trong cell
                x_cell = x * S - cell_x
                y_cell = y * S - cell_y

                target_tensor[cell_y, cell_x] = torch.tensor([1.0, x_cell, y_cell, w, h], dtype=torch.float32)

        return img, target_tensor


train_dataset2 = FistDataset(c.TRAIN_IMAGE_FOLDER, c.TRAIN_LABEL_FOLDER)
train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True, num_workers = 4)

valid_dataset2 = FistDataset(c.VALID_IMAGE_FOLDER, c.VALID_LABEL_FOLDER)
valid_loader2 = DataLoader(valid_dataset2, batch_size=32, shuffle=True, num_workers = 4)