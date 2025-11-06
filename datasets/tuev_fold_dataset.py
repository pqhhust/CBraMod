import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from scipy import signal


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            files,
    ):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.files = files

    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(os.path.join(self.data_dir, file), "rb"))
        data = data_dict['signal']
        label = int(data_dict['label'][0]-1)
        # data = signal.resample(data, 1000, axis=-1)
        data = data.reshape(16, 5, 200)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


def get_folds(subjects, seed, num_folds=4):
    """
    Split subjects into num_folds folds using the provided seed for reproducibility.
    Assumes len(subjects) is divisible by num_folds.
    """
    np.random.seed(seed)
    # Shuffle in place
    np.random.shuffle(subjects)
    fold_size = len(subjects) // num_folds
    folds = []
    for i in range(num_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < num_folds - 1 else len(subjects)
        folds.append(subjects[start:end])
    return folds


class LoadFoldDataset(object):
    def __init__(self, params, fold, seed=42):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.fold = fold
        self.seed = seed

    def get_data_loader(self):
        all_train_files = os.listdir(os.path.join(self.datasets_dir, "processed_train"))
        subjects = list(set([f.split("_")[0] for f in all_train_files]))
        subjects.sort(key=lambda x: x)
        
        folds = get_folds(subjects, self.seed)
        val_idx = self.fold - 1
        val_sub = folds[val_idx]
        train_sub = []
        for i, fold in enumerate(folds):
            if i != val_idx:
                train_sub.extend(fold)
        
        train_files = [f for f in all_train_files if f.split("_")[0] in train_sub]
        val_files = [f for f in all_train_files if f.split("_")[0] in val_sub]
        test_files = os.listdir(os.path.join(self.datasets_dir, "processed_eval"))

        train_set = CustomDataset(os.path.join(self.datasets_dir, "processed_train"), train_files)
        val_set = CustomDataset(os.path.join(self.datasets_dir, "processed_train"), val_files)
        test_set = CustomDataset(os.path.join(self.datasets_dir, "processed_eval"), test_files)

        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))

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