import scipy
from scipy import signal
import os
import re
import lmdb
import pickle
import mne
import numpy as np
from collections import defaultdict

# Define paths to the data folders (subjects)
tia_subjects = [
    '/home/user01/aiotlab/pqhung/EEG/108CMH/FA5550ZL',
    '/home/user01/aiotlab/pqhung/EEG/108CMH/FA55504C',
    '/home/user01/aiotlab/pqhung/EEG/108CMH/FA5550NQ',
    '/home/user01/aiotlab/pqhung/EEG/108CMH_C2B/FA0018QD',
    '/home/user01/aiotlab/pqhung/EEG/108CMH_C2B/FA001XWO'
]

normal_subjects = [
    '/home/user01/aiotlab/pqhung/EEG/108CMH/FA55519P',
    '/home/user01/aiotlab/pqhung/EEG/108CMH/FA55520G',
    '/home/user01/aiotlab/pqhung/EEG/108CMH/FA5550AY',
    '/home/user01/aiotlab/pqhung/EEG/108CMH/FA55517Q',
    '/home/user01/aiotlab/pqhung/EEG/108CMH/FA5551R7'
]

LABELS = { "neutral": 0, "risky": 1 }

selected_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

def folder_label_from_path(path: str) -> int:
    # Assign label based on whether the path is in tia_subjects (risky=1) or normal_subjects (neutral=0)
    for fol in tia_subjects:
        if fol in path:
            return LABELS['risky']
    for fol in normal_subjects:
        if fol in path:
            return LABELS['neutral']
    return -1

# Group subjects by label
subjects_by_label = {0: normal_subjects, 1: tia_subjects}

def generate_stratified_folds(subjects_by_label, n_folds=5, random_seed=42):
    np.random.seed(random_seed)
    folds = []
    shuffled_by_label = {}
    for lbl, subj_list in subjects_by_label.items():
        shuffled = list(subj_list)
        np.random.shuffle(shuffled)
        shuffled_by_label[lbl] = shuffled
    n_subj_per_class = len(subj_list)
    assert n_subj_per_class == n_folds, "For this setup, number of subjects per class must equal n_folds"
    for fold_id in range(n_folds):
        fold = {'train': [], 'val': [], 'test': []}
        for lbl, shuffled in shuffled_by_label.items():
            test_idx = fold_id
            val_idx = (fold_id + 1) % n_folds
            train_idxs = [i for i in range(n_folds) if i != test_idx and i != val_idx]
            fold['test'] += [shuffled[test_idx]]
            fold['val'] += [shuffled[val_idx]]
            fold['train'] += [shuffled[i] for i in train_idxs]
        # Sort for consistent output (optional)
        for k in fold:
            fold[k] = sorted(fold[k])
        folds.append(fold)
    return folds

# Generate the 5 folds
folds = generate_stratified_folds(subjects_by_label)
print("5-fold CV subject splits:")
for fid, fold in enumerate(folds):
    print(f"Fold {fid}:")
    for k in ['train', 'val', 'test']:
        subj_names = [os.path.basename(s) for s in fold[k]]
        print(f"  {k}: {len(fold[k])} subjects - {subj_names}")

# Collect all subjects
all_subjects = normal_subjects + tia_subjects
subject_keys = defaultdict(list)
all_samples = {}

# Process all subjects and collect samples in memory
for subject_dir in all_subjects:
    if not os.path.isdir(subject_dir):
        print(f'[WARN] Subject directory not found: {subject_dir}')
        continue
    subject_files = [os.path.join(subject_dir, f) for f in sorted(os.listdir(subject_dir)) if f.endswith('.edf')]
    if not subject_files:
        print(f'[WARN] No .edf files in {subject_dir}')
        continue
    for file in subject_files:
        path = file
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
        except Exception as e:
            print(f'[WARN] failed to read {path}: {e}')
            continue

        # ensure required channels exist; skip if missing
        have = set(ch.upper() for ch in raw.ch_names)
        need = [ch.upper() for ch in selected_channels]
        if not all(ch in have for ch in need):
            print(f'[WARN] missing channels in {path}, skip.')
            continue

        raw.pick(selected_channels)
        raw.reorder_channels(selected_channels)
        raw.filter(0.3, 75)
        raw.notch_filter(50)
        try:
            raw.resample(200)
        except Exception as e:
            print(f'[WARN] resample failed {path}: {e}')
            continue

        # data in microvolts as float32
        try:
            eeg = raw.get_data(units='uV', reject_by_annotation='omit').astype(np.float32)
        except TypeError:
            eeg = (raw.get_data(reject_by_annotation='omit') * 1e6).astype(np.float32)

        chs, points = eeg.shape
        a = points % (5 * 200)
        if a != 0:
            eeg = eeg[:, :-a]
        if eeg.size == 0:
            continue

        # reshape to (segments, chs, 5, 200)
        eeg = eeg.reshape(chs, -1, 5, 200).transpose(1, 0, 2, 3)

        # label per subject folder
        label = folder_label_from_path(subject_dir)
        if label == -1:
            print(f'[WARN] unknown label for {subject_dir}, skip.')
            continue

        for i, sample in enumerate(eeg):
            subject_id = os.path.basename(subject_dir)
            basename = os.path.basename(file)[:-4]  # Remove .edf extension
            sample_key = f'{subject_id}_{basename}-{i}'
            data_dict = {
                'sample': sample, 'label': label
            }
            all_samples[sample_key] = data_dict
            subject_keys[subject_id].append(sample_key)

# Generate fold datasets
fold_datasets = []
for fold_id, fold_subjs in enumerate(folds):
    fold_ds = {'train': [], 'val': [], 'test': []}
    for split_name in ['train', 'val', 'test']:
        for subj_dir in fold_subjs[split_name]:
            subj_id = os.path.basename(subj_dir)
            fold_ds[split_name].extend(subject_keys[subj_id])
    fold_datasets.append(fold_ds)

# Base directory for fold LMDBs
base_dir = '/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/tia_folds'
os.makedirs(base_dir, exist_ok=True)

# Create separate LMDB for each fold
for fold_id, fold_ds in enumerate(fold_datasets):
    db_path = os.path.join(base_dir, f'fold{fold_id}')
    db = lmdb.open(db_path, map_size=1000000000)

    # Write samples for this fold
    all_keys_in_fold = []
    for split in ['train', 'val', 'test']:
        for key in fold_ds[split]:
            if key in all_samples:
                txn = db.begin(write=True)
                txn.put(key.encode(), pickle.dumps(all_samples[key], protocol=pickle.HIGHEST_PROTOCOL))
                txn.commit()
                all_keys_in_fold.append(key)

    # Store the fold splits under '__keys__' to match CustomDataset expectation
    txn = db.begin(write=True)
    txn.put('__keys__'.encode(), pickle.dumps(fold_ds, protocol=pickle.HIGHEST_PROTOCOL))
    txn.commit()
    db.close()

    print(f"Fold {fold_id} LMDB created at {db_path} with {len(all_keys_in_fold)} samples.")

print("Preprocessing complete. 5-fold CV datasets stored in separate LMDBs.")