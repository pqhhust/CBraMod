# python finetune_main.py --downstream_dataset FACED --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Faced/processed --num_of_classes 9 --seed 42 \
# &python finetune_main.py --downstream_dataset FACED --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Faced/processed --num_of_classes 9 --seed 123 \
# &python finetune_main.py --downstream_dataset FACED --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Faced/processed --num_of_classes 9 --seed 2021 \
# &python finetune_main.py --downstream_dataset FACED --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Faced/processed --num_of_classes 9 --seed 3407 \
# &python finetune_main.py --downstream_dataset FACED --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Faced/processed --num_of_classes 9 --seed 9999

# python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --multi_lr False --seed 42 \
# &python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --multi_lr False --seed 123 \
# &python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --multi_lr False --seed 2021 \
# &python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --multi_lr False --seed 3407 \
# &python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --multi_lr False --seed 9999 \
# &wait

# python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 42 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 123 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 2021 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 3407 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 9999 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &wait

# python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 42 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/continual/epoch40_loss0.0013158333022147417.pth --frozen 1 \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 123 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/continual/epoch40_loss0.0013158333022147417.pth --frozen 1 \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 2021 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/continual/epoch40_loss0.0013158333022147417.pth --frozen 1 \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 3407 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/continual/epoch40_loss0.0013158333022147417.pth --frozen 1 \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 9999 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/continual/epoch40_loss0.0013158333022147417.pth --frozen 1 \
# &wait

# python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 42 --frozen 1 \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 123 --frozen 1 \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 2021 --frozen 1 \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 3407 --frozen 1 \
# &python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average" --num_of_classes 4 --seed 9999 --frozen 1 \
# &wait

# python finetune_main.py --downstream_dataset CHB-MIT --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/chb-mit/processed_seg" --num_of_classes 2 --seed 42 \
# &python finetune_main.py --downstream_dataset CHB-MIT --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/chb-mit/processed_seg" --num_of_classes 2 --seed 123 \
# &python finetune_main.py --downstream_dataset CHB-MIT --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/chb-mit/processed_seg" --num_of_classes 2 --seed 2021 \
# &python finetune_main.py --downstream_dataset CHB-MIT --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/chb-mit/processed_seg" --num_of_classes 2 --seed 3407 \
# &python finetune_main.py --downstream_dataset CHB-MIT --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/chb-mit/processed_seg" --num_of_classes 2 --seed 9999 \
# &wait

# python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --seed 42 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --seed 123 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --seed 2021 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --seed 3407 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --seed 9999 \
# &wait

# python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 42 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 123 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 2021 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 3407 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 9999 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch34_loss0.00140833156183362.pth \
# &wait

# python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 42 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch40_loss0.0013738306006416678.pth --frozen 1 \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 123 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch40_loss0.0013738306006416678.pth --frozen 1 \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 2021 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch40_loss0.0013738306006416678.pth --frozen 1 \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 3407 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch40_loss0.0013738306006416678.pth --frozen 1 \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 9999 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/autoformer/epoch40_loss0.0013738306006416678.pth --frozen 1 \
# &wait

# python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 42 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged/epoch40_loss0.0013504899106919765.pth \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 123 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged/epoch40_loss0.0013504899106919765.pth \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 2021 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged/epoch40_loss0.0013504899106919765.pth \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 3407 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged/epoch40_loss0.0013504899106919765.pth \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 9999 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged/epoch40_loss0.0013504899106919765.pth \
# &wait

# python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 42 --frozen 1 \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 123 --frozen 1 \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 2021 --frozen 1 \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 3407 --frozen 1 \
# &python finetune_main.py --downstream_dataset Mumtaz2016 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/MDDPHCED/processed_lmdb_75hz" --num_of_classes 2 --seed 9999 --frozen 1 \
# &wait

# python preprocessing/preprocessing_108_ica.py

# python pretrain_main.py --pretrain_dataset Merged --dataset_dir /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica_108_2022_big /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica_a7 /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica_c2b /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica_phutho /home/user01/aiotlab/pqhung/CBraMod/data/nmt_scalp_eeg_dataset /home/user01/aiotlab/pqhung/CBraMod/data/TUEG --model_dir "/home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged_ica_rescale" --seed 42

# python preprocessing/preprocessing_108_under_40.py

# python pretrain_main.py --pretrain_dataset Merged --dataset_dir /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_under40 /home/user01/aiotlab/pqhung/CBraMod/data/TUEG --model_dir "/home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged" --seed 42

# python pretrain_main.py --pretrain_dataset Merged --dataset_dir /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica --model_dir "/home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/continual" --foundation_dir "/home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/pretrained_weights.pth" --epochs 10000 --seed 42

# python continual_main.py --pretrain_dataset Merged --dataset_dir "/home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica" --foundation_dir pretrained_weights/pretrained_weights.pth --model_dir "/home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/continual" --buffer_dir "//home/user01/aiotlab/pqhung/CBraMod/data/TUEG" --seed 42

# python preprocessing/preprocessing_108_ica.py --data /home/user01/aiotlab/pqhung/EEG/108CMH_C2B --out /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica_c2b

# python preprocessing/preprocessing_108_ica.py --data /home/user01/aiotlab/pqhung/EEG/108CMH_A7 --out /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica_a7

# python preprocessing/preprocessing_phutho_ica.py --data /home/user01/aiotlab/pqhung/EEG/data_phu_tho --out /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica_phutho

# python preprocessing/preprocessing_108_ica.py --data /home/user01/aiotlab/pqhung/EEG/nmt_scalp_eeg_dataset --out /home/user01/aiotlab/pqhung/CBraMod/data/nmt_scalp_eeg_dataset

# python finetune_main.py --downstream_dataset PEARL --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/pearl/processed --num_of_classes 2 --seed 9999 --frozen 1 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged_ica_rescale/epoch40_loss0.001266965176910162.pth
# python finetune_main.py --downstream_dataset TIA --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/tia_folds/fold0 --num_of_classes 2 --seed 42 --lr 5e-4 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged_ica_rescale/epoch40_loss0.001266965176910162.pth
# python finetune_main.py --downstream_dataset TIA --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/tia_folds/fold1 --num_of_classes 2 --seed 42 --lr 5e-4 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged_ica_rescale/epoch40_loss0.001266965176910162.pth
# python finetune_main.py --downstream_dataset TIA --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/tia_folds/fold2 --num_of_classes 2 --seed 42 --lr 5e-4 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged_ica_rescale/epoch40_loss0.001266965176910162.pth
# python finetune_main.py --downstream_dataset TIA --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/tia_folds/fold3 --num_of_classes 2 --seed 42 --lr 5e-4 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged_ica_rescale/epoch40_loss0.001266965176910162.pth
# python finetune_main.py --downstream_dataset TIA --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/tia_folds/fold4 --num_of_classes 2 --seed 42 --lr 5e-4 --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged_ica_rescale/epoch40_loss0.001266965176910162.pth
# python finetune_main.py --downstream_dataset TIA --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/tia --foundation_dir /home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged_ica_rescale/epoch40_loss0.001266965176910162.pth --num_of_classes 2 --seed 42 --lr 3e-4

# python finetune_main.py --downstream_dataset BCIC-IV-2a --datasets_dir /home/user01/aiotlab/pqhung/EEG-CDCL/data/downstream/BCIC2a_processed/cbramod --num_of_classes 4 --seed 42 --foundation_dir /home/user01/aiotlab/pqhung/BIOT/log-pretrain/12-unsupervised/checkpoints/epoch=39_step=382000.ckpt --cross_validation 9

# python finetune_main.py --downstream_dataset BCIC-IV-2a --datasets_dir /home/user01/aiotlab/pqhung/EEG-CDCL/data/downstream/BCIC2a_processed/cbramod --num_of_classes 4 --seed 42 --lr 1e-3 --foundation_dir /home/user01/aiotlab/pqhung/BIOT/log-pretrain/52-unsupervised/checkpoints/epoch=38_step=46000.ckpt --cross_validation 9

# CUDA_VISIBLE_DEVICES=1 python finetune_main.py --downstream_dataset TUEV --datasets_dir ./data/datasets/BigDownstream/TUEV_refine --num_of_classes 6 --seed 42 --lr 1e-3 --foundation_dir /mnt/disk1/aiotlab/pqhung/CBraMod/data/epoch=20_step=24000.ckpt --cross_validation 4 --num_workers 8
# CUDA_VISIBLE_DEVICES=1 python finetune_main.py --downstream_dataset TUEV --datasets_dir ./data/datasets/BigDownstream/TUEV_refine --num_of_classes 6 --seed 42 --lr 1e-3 --foundation_dir /mnt/disk1/aiotlab/hieupc/New_CBraMod/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt --cross_validation 4 --num_workers 8

CUDA_VISIBLE_DEVICES=0 python finetune_main.py --downstream_dataset TUAB --datasets_dir /mnt/disk1/aiotlab/pqhung/EEG_data/TUAB/edf/process_refine --num_of_classes 2 --seed 42 --lr 1e-3 --foundation_dir /mnt/disk1/aiotlab/pqhung/CBraMod/data/epoch=20_step=24000.ckpt --cross_validation 4 --num_workers 32
# CUDA_VISIBLE_DEVICES=1 python finetune_main.py --downstream_dataset TUAB --datasets_dir /mnt/disk1/aiotlab/pqhung/EEG_data/TUAB/edf/process_refine --num_of_classes 2 --seed 42 --lr 1e-3 --foundation_dir /mnt/disk1/aiotlab/hieupc/New_CBraMod/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt --cross_validation 4 --num_workers 32