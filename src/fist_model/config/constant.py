import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #Đi ra 2 cấp tính từ file này

TRAIN_IMAGE_FOLDER = os.path.join(BASE_DIR, 'data/train/images')
TRAIN_LABEL_FOLDER = os.path.join(BASE_DIR, 'data/train/labels')
VALID_IMAGE_FOLDER = os.path.join(BASE_DIR, 'data/valid/images')
VALID_LABEL_FOLDER = os.path.join(BASE_DIR, 'data/valid/labels')


