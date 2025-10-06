import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from src.fist_model.config import constant as c

class MyDataset(Dataset):
    def __init__(self, img_folder=c.TRAIN_IMAGE_FOLDER, label_folder=c.TRAIN_LABEL_FOLDER):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.filenames = sorted(os.listdir(img_folder))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Đọc ảnh
        img_path = os.path.join(self.img_folder, self.filenames[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # C, H, W and to [0, 1]
        #--Note: Cần scale RGB về [0,1], đây là convention phổ biến. Nếu để [0, 255] thì phải chỉnh optimizer lúc train

        # Đọc label
        label_path = os.path.join(self.label_folder, self.filenames[idx].replace(".jpg", ".txt"))
        with open(label_path, "r") as f:
            nums = f.read().split()
            label = torch.tensor([1.0] + [float(x) for x in nums[:4]]) if nums else torch.zeros(5)

        return img, label

#Test----------------------------------------------------------------------
# Tạo DataLoader
train_dataset = MyDataset(c.TRAIN_IMAGE_FOLDER, c.TRAIN_LABEL_FOLDER)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers = 4)

valid_dataset = MyDataset(c.VALID_IMAGE_FOLDER, c.VALID_LABEL_FOLDER)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers = 4)

# # Dùng
# for imgs, labels in train_loader:
#     print(imgs.shape)   # (32, 3, H, W)
#     print(labels.shape) # (32, 5)
#     break

