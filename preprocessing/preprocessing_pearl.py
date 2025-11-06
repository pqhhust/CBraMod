import scipy
from scipy import signal
import os
import re
import lmdb
import pickle
import mne
import numpy as np

root_dir = '/home/user01/aiotlab/pqhung/ds004796-download'

# Only consider these subjects
NEUTRAL_SUBJECTS = [1,2,6,7,8,10,13,14,15,16,17,18,19,22,23,24,25,26,28,29,31]
RISKY_SUBJECTS   = [47,53,57,58,59,60,62,63,65,67,70,73,74,75,77,78,79,80]
LABELS = { "neutral": 0, "risky": 1 }

# collect only BrainVision task-msit_eeg .vhdr files under sub-*/eeg/
files = []
for sub in sorted([d for d in os.listdir(root_dir) if d.startswith('sub-') and os.path.isdir(os.path.join(root_dir, d))]):
    # filter to only subjects in neutral + risky sets
    m = re.match(r'^sub-(\d+)$', sub)
    if not m:
        continue
    sid = int(m.group(1))
    if (sid not in NEUTRAL_SUBJECTS) and (sid not in RISKY_SUBJECTS):
        continue
    eeg_dir = os.path.join(root_dir, sub, 'eeg')
    if not os.path.isdir(eeg_dir):
        continue
    for f in sorted(os.listdir(eeg_dir)):
        if f.endswith('_task-msit_eeg.vhdr'):
            files.append(os.path.join(sub, 'eeg', f))
files = sorted(files)
print(files)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

selected_channels = [
    'Fp1','Fp2','F3','F4','F7','F8',
    'T7','T8','C3','C4','P7','P8',
    'P3','P4','O1','O2','Fz','Cz','Pz',
]

def subject_label_from_path(rel_path: str) -> int:
    # extract subject id (sub-XX) and map via NEUTRAL/RISKY lists
    m = re.search(r'sub-(\d+)', rel_path)
    if not m:
        return -1
    sid = int(m.group(1))
    if sid in NEUTRAL_SUBJECTS:
        return LABELS['neutral']
    if sid in RISKY_SUBJECTS:
        return LABELS['risky']
    return -1

# Build a balanced (stratified) split by label with 70/10/20 ratios
files_by_label = {0: [], 1: []}
for f in files:
    lbl = subject_label_from_path(f)
    if lbl in (0, 1):
        files_by_label[lbl].append(f)

def stratified_split(files_by_label, ratios=(0.7, 0.1, 0.2), random_seed=42):
    splits = {'train': [], 'val': [], 'test': []}
    np.random.seed(random_seed)  # For reproducibility
    for lbl, lst in files_by_label.items():
        lst_shuffled = lst.copy()  # Avoid modifying original list
        np.random.shuffle(lst_shuffled)  # Randomly shuffle files
        n = len(lst_shuffled)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        n_test = n - n_train - n_val
        splits['train'] += lst_shuffled[:n_train]
        splits['val'] += lst_shuffled[n_train:n_train + n_val]
        splits['test'] += lst_shuffled[n_train + n_val:]
    # Sort for consistent output (optional, can remove if order doesn't matter)
    for k in splits:
        splits[k] = sorted(splits[k])
    return splits

files_dict = stratified_split(files_by_label, ratios=(0.7, 0.1, 0.2))
print(files_dict)

db = lmdb.open('/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/pearl/notch_filtered', map_size=1000000000)
for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        path = os.path.join(root_dir, file)
        try:
            raw = mne.io.read_raw_brainvision(path, preload=True, verbose='ERROR')
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

        # label per subject
        label = subject_label_from_path(file)
        if label == -1:
            print(f'[WARN] unknown label for {file}, skip.')
            continue

        for i, sample in enumerate(eeg):
            sample_key = f'{file[:-5]}-{i}'  # Fixed typo from file[:-4]
            data_dict = {
                'sample': sample, 'label': label
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
            txn.commit()
            dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL))
txn.commit()
db.close()