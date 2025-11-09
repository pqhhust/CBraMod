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
        data = data_dict['X']
        label = data_dict['y']
        # data = signal.resample(data, 2000, axis=-1)
        data = data.reshape(16, 10, 200)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


def get_stratified_folds(subjects, labels, seed, num_folds=4):
    """
    Split subjects into num_folds stratified folds maintaining class distribution.
    """
    from sklearn.model_selection import StratifiedKFold
    
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    # Convert to arrays for sklearn
    subjects_array = np.array(subjects)
    labels_array = np.array(labels)
    
    folds = []
    for train_idx, val_idx in skf.split(subjects_array, labels_array):
        folds.append(subjects_array[val_idx].tolist())
    
    return folds


class LoadFoldDataset(object):
    def __init__(self, params, fold, seed=42):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.fold = fold
        self.seed = seed

    def get_data_loader(self):
        all_train_files = os.listdir(os.path.join(self.datasets_dir, "train"))
        subjects = list(set([f.split("_")[0] for f in all_train_files]))
        subjects.sort(key=lambda x: x)
        
        # Get label for each subject (using first file from each subject)
        subject_labels = []
        for subject in subjects:
            subject_files = [f for f in all_train_files if f.split("_")[0] == subject]
            # Load first file to get subject's label
            data_dict = pickle.load(open(os.path.join(self.datasets_dir, "train", subject_files[0]), "rb"))
            label = data_dict['y']
            subject_labels.append(label)
        
        folds = get_stratified_folds(subjects, subject_labels, self.seed)
        val_idx = self.fold - 1
        val_sub = folds[val_idx]
        train_sub = []
        for i, fold in enumerate(folds):
            if i != val_idx:
                train_sub.extend(fold)
        
        train_files = [f for f in all_train_files if f.split("_")[0] in train_sub]
        val_files = [f for f in all_train_files if f.split("_")[0] in val_sub]
        test_files = os.listdir(os.path.join(self.datasets_dir, "test"))

        train_set = CustomDataset(os.path.join(self.datasets_dir, "train"), train_files)
        val_set = CustomDataset(os.path.join(self.datasets_dir, "train"), val_files)
        test_set = CustomDataset(os.path.join(self.datasets_dir, "test"), test_files)

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


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_files = os.listdir(os.path.join(self.datasets_dir, "train"))
        val_files = os.listdir(os.path.join(self.datasets_dir, "val"))
        test_files = os.listdir(os.path.join(self.datasets_dir, "test"))
        
        train_set = CustomDataset(os.path.join(self.datasets_dir, "train"), train_files)
        val_set = CustomDataset(os.path.join(self.datasets_dir, "val"), val_files)
        test_set = CustomDataset(os.path.join(self.datasets_dir, "test"), test_files)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
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
