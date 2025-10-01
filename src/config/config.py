from src.config.constant import TRAIN_IMAGE_FOLDER, TRAIN_LABEL_FOLDER, VALID_IMAGE_FOLDER, VALID_LABEL_FOLDER
from src.data.preprocessor import Preprocessor
import cv2
import torch
import os
train_image_path = [os.path.join(TRAIN_IMAGE_FOLDER, f) for f in os.listdir(TRAIN_IMAGE_FOLDER)]
train_label_path = [os.path.join(TRAIN_LABEL_FOLDER, f) for f in os.listdir(TRAIN_LABEL_FOLDER)]
valid_image_path = [os.path.join(VALID_IMAGE_FOLDER, f) for f in os.listdir(VALID_IMAGE_FOLDER)]
valid_label_path = [os.path.join(VALID_LABEL_FOLDER, f) for f in os.listdir(VALID_LABEL_FOLDER)]

# train_images = Preprocessor.get_data_image(train_image_path)
# train_labels = Preprocessor.get_data_label(train_label_path)
# valid_images = Preprocessor.get_data_image(valid_image_path)
valid_labels = Preprocessor.get_data_label(valid_label_path)
print(valid_labels.shape)