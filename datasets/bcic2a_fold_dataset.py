import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from collections import defaultdict

class CustomFoldDataset(Dataset):
    def __init__(
            self,
            data_dir,
            fold,
            mode='train',
    ):
        super(CustomFoldDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get('__keys__'.encode()))
        
        
        self.fold_subject = f"A{fold:02d}"  # e.g., fold=1 -> 'A01', fold=9 -> 'A09'
        prev_fold = 9 if fold == 1 else fold - 1
        self.prev_subject = f"A{prev_fold:02d}"
        
        # Filter keys based on mode
        if mode == 'test':
            self.keys = [key for key in all_keys if key.startswith(self.fold_subject)]
        elif mode == 'val':
            self.keys = [key for key in all_keys if key.startswith(self.prev_subject)]
        else:  # 'train' mode excludes the fold subject and prev subject
            self.keys = [key for key in all_keys if not key.startswith(self.fold_subject) and not key.startswith(self.prev_subject)]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()

class LoadFoldDataset(object):
    def __init__(self, params, fold):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.fold = fold

    def get_data_loader(self):
        train_set = CustomFoldDataset(self.datasets_dir, self.fold, mode='train')
        val_set = CustomFoldDataset(self.datasets_dir, self.fold, mode='val')
        test_set = CustomFoldDataset(self.datasets_dir, self.fold, mode='test')
        print(f"Fold {self.fold} - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        print(f"Total samples: {len(train_set) + len(val_set) + len(test_set)}")
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader