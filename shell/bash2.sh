# python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --seed 42 \
# &python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --seed 123 \
# &python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --seed 2021 \
# &python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --seed 3407 \
# &python finetune_main.py --downstream_dataset BCIC2020-3 --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/Imagined speech/processed" --num_of_classes 5 --lr 5e-4 --seed 9999 \
# &wait

# python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 1 --seed 42 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 1 --seed 123 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 1 --seed 2021 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 1 --seed 3407 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 1 --seed 9999 \
# &wait

# python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --epochs 100000 --seed 6 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --epochs 100000 --seed 6 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --epochs 100000 --seed 6 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --epochs 100000 --seed 6 \
# &python finetune_main.py --downstream_dataset SEED-VIG --datasets_dir "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/SEED-VIG/processed" --num_of_classes 0 --epochs 100000 --seed 6 \
# &wait

# python pretrain_main.py --pretrain_dataset Merged --dataset_dir /home/user01/aiotlab/pqhung/CBraMod/data/dummy_data_ica /home/user01/aiotlab/pqhung/CBraMod/data/TUEG --model_dir "/home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/merged" --seed 42

# python preprocessing/preprocessing_pearl.py

# python finetune_main.py --downstream_dataset PEARL --datasets_dir /home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/pearl/notch_filtered --num_of_classes 2 --seed 9999 --lr 3e-4

python preprocessing/preprocessing_tuab_biot.py