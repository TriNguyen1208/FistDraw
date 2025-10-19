import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DIR_FILE = os.path.join(BASE_DIR, 'training/data')
# FILE_NAME = [os.path.join(DIR_FILE, f) for f in os.listdir(DIR_FILE)]
# NUM_CLASSES = len(FILE_NAME)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

FILE_CLASSES_NAME = os.path.join(BASE_DIR, 'training/data.txt')
with open(FILE_CLASSES_NAME, "r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f]

NUM_CLASSES = len(CLASS_NAMES)

TRAIN_SPLIT = 0.8
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'training/model.pt')

THRESHOLD_PROB = 0.5

