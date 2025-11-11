import argparse
import random

import numpy as np
import torch

import wandb

import os

from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset, pearl_dataset, tia_dataset, bcic2a_fold_dataset, tuev_fold_dataset, tuab_fold_dataset
from finetune_trainer import Trainer
from models import model_for_faced, model_for_seedv, model_for_physio, model_for_shu, model_for_isruc, model_for_chb, \
    model_for_speech, model_for_mumtaz, model_for_seedvig, model_for_stress, model_for_tuev, model_for_tuab, \
    model_for_bciciv2a, model_for_pearl, model_for_tia, biot_for_bciciv2a, biot_for_tuev, biot_for_tuab



def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, '
                             'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a, PEARL, TIA]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/datasets/BigDownstream/Faced/processed',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='./data/wjq/models_weights/Big/BigFaced', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=False,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')
    parser.add_argument('--cross_validation', type=int, default=0, help='cross_validation')

    params = parser.parse_args()
    
    group = 'Reproduce'
    
    wandb.init(project='EEG_HUST_IP_Colab', 
               group=group,
               name=f'data_{params.downstream_dataset}_seed_{params.seed}',
               config=vars(params))
    
    print(params)

    setup_seed(params.seed)
    # torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset == 'FACED':
        load_dataset = faced_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        load_dataset = seedv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedv.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'PhysioNet-MI':
        load_dataset = physio_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_physio.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        load_dataset = shu_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_shu.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC':
        load_dataset = isruc_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        load_dataset = chb_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_chb.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        load_dataset = speech_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_speech.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'Mumtaz2016':
        load_dataset = mumtaz_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mumtaz.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'SEED-VIG':
        load_dataset = seedvig_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedvig.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_regression()
    elif params.downstream_dataset == 'MentalArithmetic':
        load_dataset = stress_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_stress.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        if params.cross_validation != 0:
            for fold in range(1, params.cross_validation + 1):
                print(f'Cross-validation fold {fold}')
                load_dataset = tuev_fold_dataset.LoadFoldDataset(params, fold)
                data_loader = load_dataset.get_data_loader()
                model = biot_for_tuev.Model(params)
                t = Trainer(params, data_loader, model)
                t.train_for_multiclass_fold(fold)
        else:
            load_dataset = tuev_dataset.LoadDataset(params)
            data_loader = load_dataset.get_data_loader()
            model = model_for_tuev.Model(params)
            t = Trainer(params, data_loader, model)
            t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        if params.cross_validation != 0:
            for fold in range(3, params.cross_validation + 1):
                print(f'Cross-validation fold {fold}')
                load_dataset = tuab_fold_dataset.LoadFoldDataset(params, fold)
                data_loader = load_dataset.get_data_loader()
                model = biot_for_tuab.Model(params)
                t = Trainer(params, data_loader, model)
                t.train_for_binaryclass_fold(fold)
        else:
            load_dataset = tuab_dataset.LoadDataset(params)
            data_loader = load_dataset.get_data_loader()
            model = model_for_tuab.Model(params)
            t = Trainer(params, data_loader, model)
            t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC-IV-2a':
        if params.cross_validation != 0:
            for fold in range(1, params.cross_validation + 1):
                print(f'Cross-validation fold {fold}')
                load_dataset = bcic2a_fold_dataset.LoadFoldDataset(params, fold)
                data_loader = load_dataset.get_data_loader()
                model = biot_for_bciciv2a.Model(params)
                t = Trainer(params, data_loader, model)
                t.train_for_multiclass_fold(fold)
        else:   
            load_dataset = bciciv2a_dataset.LoadDataset(params) 
            data_loader = load_dataset.get_data_loader()
            model = model_for_bciciv2a.Model(params)
            t = Trainer(params, data_loader, model)
            t.train_for_multiclass()
    elif params.downstream_dataset == 'PEARL':
        load_dataset = pearl_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_pearl.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TIA':
        load_dataset = tia_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tia.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    print('Done!!!!!')


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    wandb.login(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')
    main()
    wandb.finish()
