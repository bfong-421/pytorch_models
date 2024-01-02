import pandas as pd
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split

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


class YelpReviewPolarity(Dataset):
    def __init__(self, root, split='train'):
        super(YelpReviewPolarity, self).__init__()
        filename = None
        frame = None
        if split == 'train':
            filename = 'train.csv'
        elif split == 'test':
            filename = 'test.csv'
        try:
            frame = pd.read_csv(os.path.join(root, filename))
        except TypeError:
            print('Invalid split')
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, item):
        text = self.frame.iloc[item, 1]
        label = self.frame.iloc[item, 0]
        return text, label


class AG_NEWS(Dataset):
    def __init__(self, root, split='train'):
        super(AG_NEWS, self).__init__()
        frame = None
        filename = None
        if split == 'train':
            filename = 'train.csv'
        elif split == 'test':
            filename = 'test.csv'
        try:
            frame = pd.read_csv(os.path.join(root, filename))
        except TypeError:
            print('Invalid split')
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, item):
        text = self.frame.iloc[item, 0]
        label = self.frame.iloc[item, 1]
        return text, label


class DBpedia(Dataset):
    def __init__(self, root, split='train'):
        super(DBpedia, self).__init__()
        frame = None
        filename = None
        if split == 'train':
            filename = 'DBPEDIA_train.csv'
        elif split == 'test':
            filename = 'DBPEDIA_test.csv'
        elif split == 'val':
            filename = 'DBPEDIA_val.csv'
        try:
            frame = pd.read_csv(os.path.join(root, filename))[['text', 'l1']]
        except TypeError:
            print('Invalid split')
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, item):
        text = self.frame.iloc[item, 0]
        label = self.frame.iloc[item, 1]
        return text, label


class NewsAggregator(Dataset):
    def __init__(self, root, split='train'):
        super(NewsAggregator, self).__init__()
        frame = pd.read_csv(os.path.join(root, 'uci-news-aggregator.csv'))[['TITLE', 'CATEGORY']]
        test_size = 0.2
        val_size = 0.1
        train_split, temp_split = train_test_split(frame, test_size=(test_size + val_size), random_state=42)
        test_split, val_split = train_test_split(temp_split, test_size=val_size/(test_size + val_size), random_state=42)
        if split == 'train':
            frame = train_split
        elif split == 'test':
            frame = test_split
        elif split == 'val':
            frame = val_split
        else:
            raise TypeError('Invalid split')
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, item):
        text = self.frame.iloc[item, 0]
        label = self.frame.iloc[item, 1]
        return text, label

