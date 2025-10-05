import torch
import os
from src.draw_model.data.preprocessing import Preprocessing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_FILE = os.path.join(BASE_DIR, 'data/dataset')
FILE_NAME = [os.path.join(DIR_FILE, f) for f in os.listdir(DIR_FILE)]
NUM_CLASSES = len(FILE_NAME)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CLASS_NAMES = [os.path.splitext(f)[0] for f in os.listdir(DIR_FILE)]

TRAIN_SPLIT = 0.8
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'training/model.pkl')
