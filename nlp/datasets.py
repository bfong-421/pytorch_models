import pandas as pd
from torch.utils.data import Dataset
import os

class AmazonReviewHierarchical(Dataset):
    def __init__(self, root, split='train'):
        super().__init__()
        filename = None
        frame = None
        if split == 'train':
            filename = 'train_40k.csv'
        elif split == 'val':
            filename = 'val_10k.csv'
        try:
            frame = pd.read_csv(os.path.join(root, filename))[['Text', 'Cat1']]
        except TypeError:
            print('Invalid split')
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, item):
        text = self.frame.iloc[item, 0]
        label = self.frame.iloc[item, 1]
        return text, label
