import os
import pickle

from multiprocessing import Pool
import numpy as np
import mne

channelMap = {
  "EEG FP1-REF": "Fp1",
  "EEG F7-REF":  "F7",
  "EEG T3-REF":  "T3",
  "EEG T5-REF":  "T5",
  "EEG O1-REF":  "O1",
  "EEG FP2-REF": "Fp2",
  "EEG F8-REF":  "F8",
  "EEG T4-REF":  "T4",
  "EEG T6-REF":  "T6",
  "EEG O2-REF":  "O2",
  "EEG F3-REF":  "F3",
  "EEG C3-REF":  "C3",
  "EEG P3-REF":  "P3",
  "EEG F4-REF":  "F4",
  "EEG C4-REF":  "C4",
  "EEG P4-REF":  "P4"
}

def split_and_dump(params):
    """
    lọc file theo bệnh nhân → load & preprocess → tính montage bipolar → cắt cửa sổ 2000 mẫu → lưu từng cửa sổ kèm nhãn
    Args:
        fetch_folder:  folder chứa các file .edf
        sub: tên bệnh nhân
        dump_folder: folder ghi ra file .pkl
        label: nhãn cho dữ liệu
    """
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if sub in file: # Lọc file theo bệnh nhân
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True) # Dùng MNE để load toàn bộ dữ liệu EDF
            raw.resample(200) # Resample xuống 200 Hz
            raw.filter(l_freq=0.3, h_freq=75) # Band-pass filter từ 0.3–75 Hz
            raw.notch_filter((60)) # Notch filter tần số 60 Hz
            ch_name = raw.ch_names
            raw_data = raw.get_data(units='uV') # Đọc toàn bộ mảng tín hiệu
            channeled_data = raw_data.copy()[:16]
            try: # Tính các kênh bipolar
                channeled_data[0] = (
                    raw_data[ch_name.index(channelMap["EEG FP1-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG F7-REF"])]
                )
                channeled_data[1] = (
                    raw_data[ch_name.index(channelMap["EEG F7-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG T3-REF"])]
                )
                channeled_data[2] = (
                    raw_data[ch_name.index(channelMap["EEG T3-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG T5-REF"])]
                )
                channeled_data[3] = (
                    raw_data[ch_name.index(channelMap["EEG T5-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG O1-REF"])]
                )
                channeled_data[4] = (
                    raw_data[ch_name.index(channelMap["EEG FP2-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG F8-REF"])]
                )
                channeled_data[5] = (
                    raw_data[ch_name.index(channelMap["EEG F8-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG T4-REF"])]
                )
                channeled_data[6] = (
                    raw_data[ch_name.index(channelMap["EEG T4-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG T6-REF"])]
                )
                channeled_data[7] = (
                    raw_data[ch_name.index(channelMap["EEG T6-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG O2-REF"])]
                )
                channeled_data[8] = (
                    raw_data[ch_name.index(channelMap["EEG FP1-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG F3-REF"])]
                )
                channeled_data[9] = (
                    raw_data[ch_name.index(channelMap["EEG F3-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG C3-REF"])]
                )
                channeled_data[10] = (
                    raw_data[ch_name.index(channelMap["EEG C3-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG P3-REF"])]
                )
                channeled_data[11] = (
                    raw_data[ch_name.index(channelMap["EEG P3-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG O1-REF"])]
                )
                channeled_data[12] = (
                    raw_data[ch_name.index(channelMap["EEG FP2-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG F4-REF"])]
                )
                channeled_data[13] = (
                    raw_data[ch_name.index(channelMap["EEG F4-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG C4-REF"])]
                )
                channeled_data[14] = (
                    raw_data[ch_name.index(channelMap["EEG C4-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG P4-REF"])]
                )
                channeled_data[15] = (
                    raw_data[ch_name.index(channelMap["EEG P4-REF"])]
                    - raw_data[ch_name.index(channelMap["EEG O2-REF"])]
                )
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            # chia dải tín hiệu thành các cửa sổ dài 2000 mẫu
            for i in range(channeled_data.shape[1] // 2000): 
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                # Dump ra file .pkl
                pickle.dump(
                    {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
                    open(dump_path, "wb"),
                )
                
if __name__ == "__main__":
    # root to abnormal dataset
    root = "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/108CMH"
    edf = "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/108CMH/edf"
    channel_std = "01_tcp_ar"

    # seed = 4523
    # np.random.seed(seed)
    data = list(
        set([item for item in os.listdir(edf)])
    )
    train, val, test = (
        data[: int(len(data) * 0.8)],
        data[int(len(data) * 0.8) : int(len(data) * 0.9)],
        data[int(len(data) * 0.9) :],
    )
    
    # # train, val abnormal subjects
    # train_val_abnormal = os.path.join(root, "train", "abnormal", channel_std)
    # train_val_a_sub = list(
    #     set([item.split("_")[0] for item in os.listdir(train_val_abnormal)])
    # )
    # train_val_a_sub.sort(key=lambda x: x)

    # train_a_sub, val_a_sub = (
    #     train_val_a_sub[: int(len(train_val_a_sub) * 0.8)],
    #     train_val_a_sub[int(len(train_val_a_sub) * 0.8) :],
    # )
    # print('train_a_sub:', train_a_sub)
    # print('val_a_sub:', val_a_sub)

    # # train, val normal subjects
    # train_val_normal = os.path.join(root, "train", "normal", channel_std)
    # train_val_n_sub = list(
    #     set([item.split("_")[0] for item in os.listdir(train_val_normal)])
    # )
    # train_val_n_sub.sort(key=lambda x: x)

    # train_n_sub, val_n_sub = (
    #     train_val_n_sub[: int(len(train_val_n_sub) * 0.8)],
    #     train_val_n_sub[int(len(train_val_n_sub) * 0.8) :],
    # )
    # print('train_n_sub:', train_n_sub)
    # print('val_n_sub:', val_n_sub)


    # # test abnormal subjects
    # test_abnormal = os.path.join(root, "eval", "abnormal", channel_std)
    # test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))

    # # test normal subjects
    # test_normal = os.path.join(root, "eval", "normal", channel_std)
    # test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))

    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(root, "process_refine")):
        os.makedirs(os.path.join(root, "process_refine"))

    if not os.path.exists(os.path.join(root, "process_refine", "train")):
        os.makedirs(os.path.join(root, "process_refine", "train"))
    train_dump_folder = os.path.join(root, "process_refine", "train")

    if not os.path.exists(os.path.join(root, "process_refine", "val")):
        os.makedirs(os.path.join(root, "process_refine", "val"))
    val_dump_folder = os.path.join(root, "process_refine", "val")

    if not os.path.exists(os.path.join(root, "process_refine", "test")):
        os.makedirs(os.path.join(root, "process_refine", "test"))
    test_dump_folder = os.path.join(root, "process_refine", "test")

    # fetch_folder, sub, dump_folder, labels
    # parameters = []
    # for train_sub in train_a_sub:
    #     parameters.append([train_val_abnormal, train_sub, train_dump_folder, 1])
    # for train_sub in train_n_sub:
    #     parameters.append([train_val_normal, train_sub, train_dump_folder, 0])
    # for val_sub in val_a_sub:
    #     parameters.append([train_val_abnormal, val_sub, val_dump_folder, 1])
    # for val_sub in val_n_sub:
    #     parameters.append([train_val_normal, val_sub, val_dump_folder, 0])
    # for test_sub in test_a_sub:
    #     parameters.append([test_abnormal, test_sub, test_dump_folder, 1])
    # for test_sub in test_n_sub:
    #     parameters.append([test_normal, test_sub, test_dump_folder, 0])

    # split and dump in parallel
    with Pool(processes=24) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)

    print('Done!')
    