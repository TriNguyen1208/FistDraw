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
        label_path = os.path.join(self.label_folder, self.files[idx].replace('.jpg','.txt'))
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h0,w0,_ = img.shape

        img = cv2.resize(img,(224,224))
        img = torch.from_numpy(img).permute(2,0,1).float()/255.0

        # read label
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            nums = [float(x) for x in open(label_path).read().split()]
            target = torch.tensor([1.0]+nums[:4], dtype=torch.float32)  # p=1
        else:
            target = torch.zeros(5, dtype=torch.float32)

        return img, target

train_dataset2 = FistDataset(c.TRAIN_IMAGE_FOLDER, c.TRAIN_LABEL_FOLDER)
train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True, num_workers = 4)

valid_dataset2 = FistDataset(c.VALID_IMAGE_FOLDER, c.VALID_LABEL_FOLDER)
valid_loader2 = DataLoader(valid_dataset2, batch_size=32, shuffle=False, num_workers = 4)