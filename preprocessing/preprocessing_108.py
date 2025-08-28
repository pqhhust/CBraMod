import os
import random

import mne
import numpy as np
from tqdm import tqdm
import pickle
import lmdb

selected_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def iter_files(rootDir):
    file_path_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            # print(file_name)
            file_path_list.append(file_name)
    return file_path_list

def preprocessing_recording(file_path, file_key_list: list, db):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    raw.pick_channels(selected_channels, ordered=True)
    # print(raw.info)
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((50))
    eeg_array = raw.to_data_frame().values
    # print(raw.info)
    eeg_array = eeg_array[:, 1:]
    points, chs = eeg_array.shape
    # Segment into 30s windows at 200 Hz without dropping any data.
    # Keep all samples by zero-padding the final window if needed.
    L = 30 * 200
    if points <= 0:
        return
    full = points // L
    segments = []
    for i in range(full):
        segments.append(eeg_array[i * L : (i + 1) * L, :])
    rem = points - full * L
    if rem > 0:
        last = np.zeros((L, chs), dtype=eeg_array.dtype)
        last[:rem, :] = eeg_array[full * L :, :]
        segments.append(last)
    eeg_array = np.stack(segments, axis=0)  # (n_win, L, chs)
    eeg_array = eeg_array.reshape(-1, 30, 200, chs)
    eeg_array = eeg_array.transpose(0, 3, 1, 2)  # (n_win, chs, 30, 200)
    # print(eeg_array.shape)
    file_name = file_path.split('/')[-1][:-4]

    for i, sample in enumerate(eeg_array):
        # print(i, sample.shape)
        sample_key = f'{file_name}_{i}'
        print(sample_key)
        file_key_list.append(sample_key)
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(sample))
        txn.commit()

if __name__ == '__main__':
    setup_seed(1)
    file_path_list = iter_files('/mnt/disk1/pqhung/EEG/edf_files')

    file_path_list = sorted(file_path_list)
    random.shuffle(file_path_list)
    # print(file_path_list)
    db = lmdb.open(r'/mnt/disk1/pqhung/CBraMod/data/108CMH', map_size=17179869184)
    file_key_list = []
    for file_path in tqdm(file_path_list):
        preprocessing_recording(file_path, file_key_list, db)

    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(file_key_list))
    txn.commit()
    db.close()
