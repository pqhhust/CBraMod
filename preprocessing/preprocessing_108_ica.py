import os
import random

import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs , create_eog_epochs
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

def ICA_for_raw(raw_full):
    raw_for_ica = raw_full.copy()
    duration = raw_for_ica.times[-1]
    tmax = min(duration, 60) 
    raw_for_ica.crop(tmax=tmax)

    filt_raw = raw_for_ica.filter(l_freq=0.5, h_freq=None)

    ica = ICA(n_components=10, max_iter="auto", random_state=42)
    ica.fit(filt_raw)

    eog_indices, _ = ica.find_bads_eog(raw_for_ica, ch_name='Fp1', threshold=0.9)

    ecg_indices, _ = ica.find_bads_ecg(raw_for_ica, ch_name='C3', method='correlation', threshold=0.9)

    ica.exclude = ecg_indices + eog_indices

    reconst_raw = raw_full.copy()
    ica.apply(reconst_raw)

    return reconst_raw

def preprocessing_recording(file_path, file_key_list: list, db):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    raw.pick_channels(selected_channels, ordered=True)
    raw = ICA_for_raw(raw)
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
        if np.max(np.abs(sample)) < 100:
            sample_key = f'{file_name}_{i}'
            print(sample_key)
            file_key_list.append(sample_key)
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(sample))
            txn.commit()
        # else:
        #     # Rescale sample to [-100, 100]
        #     fixed_sample = sample.copy()
        #     min_val = fixed_sample.min()
        #     max_val = fixed_sample.max()
        #     if max_val != min_val:
        #         fixed_sample = 200 * (fixed_sample - min_val) / (max_val - min_val) - 100
        #     else:
        #         fixed_sample[:] = 0  # If constant, set to 0
        #     sample_key = f'{file_name}_{i}_rescaled'
        #     print(f"{sample_key} (rescaled)")
        #     file_key_list.append(sample_key)
        #     txn = db.begin(write=True)
        #     txn.put(key=sample_key.encode(), value=pickle.dumps(fixed_sample))
        #     txn.commit()          

if __name__ == '__main__':
    setup_seed(1)
    file_path_list = iter_files('/home/user01/aiotlab/pqhung/EEG/108CMH')

    file_path_list = sorted(file_path_list)
    random.shuffle(file_path_list)
    # print(file_path_list)
    db = lmdb.open(r'/home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica_rescale', map_size=17179869184)
    file_key_list = []
    for file_path in tqdm(file_path_list):
        preprocessing_recording(file_path, file_key_list, db)

    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(file_key_list))
    txn.commit()
    db.close()
