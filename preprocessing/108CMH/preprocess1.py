import os, glob
import numpy as np
import pyedflib

def merge_edfs(edf_paths, out_path):
    # Mở file đầu để lấy header
    reader0 = pyedflib.EdfReader(edf_paths[0])
    n_ch = reader0.signals_in_file
    freqs = [reader0.getSampleFrequency(i) for i in range(n_ch)]
    hdrs  = [reader0.getSignalHeader(i)    for i in range(n_ch)]
    reader0.close()

    # Đọc và nối dữ liệu kênh-wise
    data = []
    for ch in range(n_ch):
        segments = []
        for p in edf_paths:
            r = pyedflib.EdfReader(p)
            segments.append(r.readSignal(ch))
            r.close()
        data.append(np.concatenate(segments))

    # Ghi file EDF mới
    w = pyedflib.EdfWriter(out_path, n_ch)
    # Thiết lập tần số mẫu cho từng kênh
    for ch, sf in enumerate(freqs):
        w.setSamplefrequency(ch, sf)
    # Thiết lập header cho từng kênh
    w.setSignalHeaders(hdrs)
    
    # Ghi dữ liệu
    w.writeSamples(data)
    w.close()
    print(f"✔ Merged {len(edf_paths)} → {out_path}")

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_edfs = glob.glob(os.path.join(input_dir, "*.edf"))

    # Nhóm theo phần trước dấu "-" cuối trong tên file
    groups = {}
    for f in all_edfs:
        base   = os.path.splitext(os.path.basename(f))[0]
        prefix = base.rsplit('-', 1)[0]
        groups.setdefault(prefix, []).append(f)

    for prefix, files in groups.items():
        files.sort()
        out_path = os.path.join(output_dir, prefix + ".edf")
        merge_edfs(files, out_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Merge các segment EDF của từng bệnh nhân thành 1 EDF duy nhất"
    )
    p.add_argument("--input_dir",  help="Thư mục chứa các file .edf gốc")
    p.add_argument("--output_dir", help="Thư mục lưu các file đã merge")
    args = p.parse_args()

    main(args.input_dir, args.output_dir)