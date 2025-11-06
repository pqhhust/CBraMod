import scipy
from scipy import signal
import os
import re
import lmdb
import pickle
import mne
import numpy as np

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

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

# selected_channels = [
#     'Fp1','Fp2','F3','F4','F7','F8',
#     'T7','T8','C3','C4','P7','P8',
#     'P3','P4','O1','O2','Fz','Cz','Pz',
# ]

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

# Build a balanced (stratified) split by subject/label with 70/10/20 ratios
def stratified_split_subjects(subjects_by_label, ratios=(0.6, 0.2, 0.2), random_seed=42):
    splits = {'train': [], 'val': [], 'test': []}
    np.random.seed(random_seed)  # For reproducibility
    for lbl, subj_list in subjects_by_label.items():
        subj_shuffled = subj_list.copy()  # Avoid modifying original list
        np.random.shuffle(subj_shuffled)  # Randomly shuffle subjects
        n = len(subj_shuffled)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        n_test = n - n_train - n_val
        splits['train'] += subj_shuffled[:n_train]
        splits['val'] += subj_shuffled[n_train:n_train + n_val]
        splits['test'] += subj_shuffled[n_train + n_val:]
    # Sort for consistent output (optional, can remove if order doesn't matter)
    for k in splits:
        splits[k] = sorted(splits[k])
    return splits

subjects_dict = stratified_split_subjects(subjects_by_label, ratios=(0.6, 0.2, 0.2))
print("Subject splits by set:")
for k, v in subjects_dict.items():
    print(f"{k}: {len(v)} subjects - {v}")

# Open LMDB database (adjust path if needed for a new dataset)
db = lmdb.open('/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/tia/', map_size=1000000000)
for files_key in subjects_dict.keys():
    for subject_dir in subjects_dict[files_key]:
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
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
                txn.commit()
                dataset[files_key].append(sample_key)

# Store the dataset keys
txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL))
txn.commit()
db.close()

print("Preprocessing complete. Dataset stored in LMDB.")