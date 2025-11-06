import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pretraining_dataset import PretrainingDataset
from datasets.merged_dataset import MergedPretrainingDataset
from models import cbramod
from pretrain_trainer import Trainer, CLTrainer

import wandb

import copy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--parallel', type=bool, default=False, help='parallel')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='lr_scheduler: CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, CyclicLR')

    # parser.add_argument('--project_mode', type=str, default='cnn', help='project_mode')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    parser.add_argument('--need_mask', type=bool, default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')
    
    parser.add_argument('--cl_epochs', type=int, default=40, help='number of epochs for cl (default: 5)')
    parser.add_argument('--cl_batch_size', type=int, default=128, help='batch size for training for cl (default: 32)')
    parser.add_argument('--cl_lr', type=float, default=1e-4, help='learning rate for cl (default: 1e-3)')
    parser.add_argument('--cl_weight_decay', type=float, default=5e-2, help='weight_decay for cl')
    parser.add_argument('--cl_clip_value', type=float, default=1, help='clip_value for cl')
    parser.add_argument('--cl_lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='lr_scheduler: CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, CyclicLR for cl')
    parser.add_argument('--align_every', type=int, default=2, help='align_every for cl')
    parser.add_argument('--buffer_dir', type=str, default=None, help='buffer_dir for cl')

    parser.add_argument('--pretrain_dataset', type=str, default='TUEG',
                        help='[TUEG, Merged]')
    
    parser.add_argument('--foundation_dir', type=str,
                        default=None,
                        help='foundation_dir')

    parser.add_argument('--dataset_dir', nargs='+', required=True,
                        help='one or more LMDB dataset directories; if TUEG, the first is used; if Merged, all are used')
    parser.add_argument('--model_dir',   type=str,   default='model_dir', help='model_dir')
    params = parser.parse_args()
    
    group = 'Continual-Transformer'
    
    wandb.init(project='EEG_HUST_IP_Colab', 
               group=group,
               name=f'pretrain_seed_{params.seed}',
               config=vars(params))
    
    print(params)
    setup_seed(params.seed)
    if params.pretrain_dataset == 'TUEG':
        if not params.dataset_dir or len(params.dataset_dir) == 0:
            raise ValueError('Please provide at least one --dataset_dir for TUEG')
        pretrained_dataset = PretrainingDataset(dataset_dir=params.dataset_dir[0])
    elif params.pretrain_dataset == 'Merged':
        dirs = params.dataset_dir
        if not dirs or any(d is None or d == '' for d in dirs):
            raise ValueError('Please provide one or more valid --dataset_dir values for Merged')
        pretrained_dataset = MergedPretrainingDataset(dataset_dirs=dirs)
    print(len(pretrained_dataset))
    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=4,
        shuffle=True,
    )
    model = cbramod.CBraMod(
        params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
        params.nhead
    )
    if params.foundation_dir:
        map_location = torch.device(f'cuda:{params.cuda}')
        model.load_state_dict(torch.load(params.foundation_dir, map_location=map_location))
    trainer = Trainer(params, data_loader, model)
    trainer.train()
    
    params.epochs = params.cl_epochs
    params.batch_size = params.cl_batch_size
    params.lr = params.cl_lr
    params.weight_decay = params.cl_weight_decay
    params.clip_value = params.cl_clip_value
    params.lr_scheduler = params.cl_lr_scheduler

    buffer_loader = DataLoader(
        PretrainingDataset(dataset_dir=params.buffer_dir),
        batch_size=params.cl_batch_size,
        num_workers=4,
        shuffle=True,
    )

    ref_model = copy.deepcopy(model)
    model.load_state_dict(torch.load(params.foundation_dir, map_location='cpu'))
    cl_trainer = CLTrainer(params, data_loader, model, ref_model, buffer_loader)
    cl_trainer.train()
    
    db = getattr(pretrained_dataset, 'db', None)
    if db is not None:
        try:
            db.close()
        except Exception:
            pass
    dbs = getattr(pretrained_dataset, 'dbs', None)
    if dbs is not None:
        for _db in dbs:
            try:
                _db.close()
            except Exception:
                pass


if __name__ == '__main__':
    wandb.login(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')
    main()
    wandb.finish()
