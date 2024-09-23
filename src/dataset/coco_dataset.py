from torch.utils.data import Dataset
import torch
import numpy as np
from config import *


class CocoDataset(Dataset):
    def __init__(self, split: str):
        if split == SPLIT_TRAIN:
            with open(TRAIN_ANNO_PATH, 'r') as f:
                anno_json = json.load(f)
