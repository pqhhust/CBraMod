import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.nn import MSELoss
from torchinfo import summary
from tqdm import tqdm

import os

from utils.util import generate_mask

import wandb

import copy


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction='mean').to(self.device)

        if self.params.parallel:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)

        summary(self.model, input_size=(1, 19, 30, 200))

        macs, params = get_model_complexity_info(self.model, (19, 30, 200), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                           weight_decay=self.params.weight_decay)

        if self.params.lr_scheduler=='CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler=='ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif self.params.lr_scheduler=='StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5*self.data_length, gamma=0.5
            )
        elif self.params.lr_scheduler=='MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10*self.data_length, 20*self.data_length, 30*self.data_length], gamma=0.1
            )
        elif self.params.lr_scheduler=='CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=0.001, step_size_up=self.data_length*5,
                step_size_down=self.data_length*2, mode='exp_range', gamma=0.9, cycle_momentum=False
            )


    def train(self):
        if not os.path.exists(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        best_loss = 10000
        for epoch in range(self.params.epochs):
            losses = []
            for x in tqdm(self.data_loader, mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)/100
                if self.params.need_mask:
                    bz, ch_num, patch_num, patch_size = x.shape
                    mask = generate_mask(
                        bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device,
                    )
                    y = self.model(x, mask=mask)
                    masked_x = x[mask == 1]
                    masked_y = y[mask == 1]
                    loss = self.criterion(masked_y, masked_x)

                    # non_masked_x = x[mask == 0]
                    # non_masked_y = y[mask == 0]
                    # non_masked_loss = self.criterion(non_masked_y, non_masked_x)
                    # loss = 0.8 * masked_loss + 0.2 * non_masked_loss
                else:
                    y = self.model(x)
                    loss = self.criterion(y, x)
                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.data.cpu().numpy())
            mean_loss = np.mean(losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": mean_loss,
                "lr": learning_rate,
            }, step = epoch + 1)
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.6f}')
            if  mean_loss < best_loss:
                model_path = rf'{self.params.model_dir}/epoch{epoch+1}_loss{mean_loss}.pth'
                torch.save(self.model.state_dict(), model_path)
                print("model save in " + model_path)
                best_loss = mean_loss
                
class CLTrainer(object):
    def __init__(self, params, data_loader, model, ref_model, buffer_loader):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.buffer_loader = buffer_loader
        self.model = model.to(self.device)
        self.ref_model = ref_model.to(self.device)
        self.criterion = MSELoss(reduction='mean').to(self.device)

        if self.params.parallel:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)

        summary(self.model, input_size=(1, 19, 30, 200))

        macs, params = get_model_complexity_info(self.model, (19, 30, 200), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                           weight_decay=self.params.weight_decay)

        if self.params.lr_scheduler=='CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler=='ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif self.params.lr_scheduler=='StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5*self.data_length, gamma=0.5
            )
        elif self.params.lr_scheduler=='MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10*self.data_length, 20*self.data_length, 30*self.data_length], gamma=0.1
            )
        elif self.params.lr_scheduler=='CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=0.001, step_size_up=self.data_length*5,
                step_size_down=self.data_length*2, mode='exp_range', gamma=0.9, cycle_momentum=False
            )


    def train(self):
        if not os.path.exists(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        best_loss = 10000
        # Use underlying module for feature-level calls if DataParallel
        model_for_feat = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        current_model = copy.deepcopy(model_for_feat).to(self.device)
        current_model.eval()
        for epoch in range(self.params.epochs):
            losses = []
            masked_losses = []
            distill_losses = []
            buffer_losses = []
            align_losses = []
            buffer_iter = iter(self.buffer_loader)
            for x in tqdm(self.data_loader, mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)/100
                # Safe defaults so we can always log
                masked_loss = torch.tensor(0.0, device=self.device)
                distill_loss = torch.tensor(0.0, device=self.device)
                buffer_loss = torch.tensor(0.0, device=self.device)
                align_loss = torch.tensor(0.0, device=self.device)
                if self.params.need_mask:
                    ## traditional masked loss
                    bz, ch_num, patch_num, patch_size = x.shape
                    mask = generate_mask(
                        bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device,
                    )
                    feature = model_for_feat.patch_embedding(x, mask)
                    feature = model_for_feat.encoder(feature)
                    y = model_for_feat.proj_out(feature)
                    # y = self.model(x, mask=mask)
                    masked_x = x[mask == 1]
                    masked_y = y[mask == 1]
                    masked_loss = self.criterion(masked_y, masked_x)

                    ## distillation loss
                    with torch.no_grad():
                        ref_backbone = self.ref_model.module if isinstance(self.ref_model, torch.nn.DataParallel) else self.ref_model
                        ref_feature = ref_backbone.patch_embedding(x, mask)
                        ref_feature = ref_backbone.encoder(ref_feature)
                    distill_loss = torch.mean((feature - ref_feature) ** 2)

                    ## buffer loss (apply masking to buffer samples)
                    try:
                        x_buffer = next(buffer_iter)
                    except StopIteration:
                        buffer_iter = iter(self.buffer_loader)
                        x_buffer = next(buffer_iter)
                    x_buffer = x_buffer.to(self.device) / 100
                    bz_b, ch_b, patch_b, size_b = x_buffer.shape
                    mask_buffer = generate_mask(
                        bz_b, ch_b, patch_b, mask_ratio=self.params.mask_ratio, device=self.device,
                    )
                    feature_buffer = model_for_feat.patch_embedding(x_buffer, mask=mask_buffer)
                    feature_buffer = model_for_feat.encoder(feature_buffer)
                    y_buffer = model_for_feat.proj_out(feature_buffer)
                    masked_x_buffer = x_buffer[mask_buffer == 1]
                    masked_y_buffer = y_buffer[mask_buffer == 1]
                    buffer_loss = self.criterion(masked_y_buffer, masked_x_buffer)

                    ## buffer alignment loss
                    align_loss = torch.tensor(0.0).to(self.device)
                    if getattr(self.params, 'align_every', 0) and (epoch + 1) % self.params.align_every == 0:
                        with torch.no_grad():
                            cur_feature = current_model.patch_embedding(x_buffer, mask=mask_buffer)
                            cur_feature = current_model.encoder(cur_feature)
                            # cur_y = current_model.proj_out(cur_feature)
                            cur_vec = cur_feature.reshape((bz_b, -1))
                        buf_vec = feature_buffer.reshape((bz_b, -1))

                        cos_dist = 1 - torch.nn.functional.cosine_similarity(cur_vec, buf_vec, dim=-1)
                        align_loss = cos_dist.mean()

                    # Weighted loss aggregation
                    mask_weight = getattr(self.params, 'mask_weight', 1.0)
                    distill_weight = getattr(self.params, 'distill_weight', 1.0)
                    buffer_weight = getattr(self.params, 'buffer_weight', 1.0)
                    align_weight = getattr(self.params, 'align_weight', 1.0)
                    loss = (
                        mask_weight * masked_loss
                        + distill_weight * distill_loss
                        + buffer_weight * buffer_loss
                        + align_weight * align_loss
                    )

                    # non_masked_x = x[mask == 0]
                    # non_masked_y = y[mask == 0]
                    # non_masked_loss = self.criterion(non_masked_y, non_masked_x)
                    # loss = 0.8 * masked_loss + 0.2 * non_masked_loss
                else:
                    y = self.model(x)
                    loss = self.criterion(y, x)
                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.data.cpu().numpy())
                masked_losses.append(masked_loss.data.cpu().numpy())
                distill_losses.append(distill_loss.data.cpu().numpy())
                buffer_losses.append(buffer_loss.data.cpu().numpy())
                align_losses.append(align_loss.data.cpu().numpy())

            if getattr(self.params, 'align_every', 0) and (epoch + 1) % self.params.align_every == 0:
                src = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                current_model.load_state_dict(src.state_dict())
            mean_loss = np.mean(losses)
            mean_masked_loss = np.mean(masked_losses)
            mean_distill_loss = np.mean(distill_losses)
            mean_buffer_loss = np.mean(buffer_losses)
            mean_align_loss = np.mean(align_losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            wandb.log({
                "cl_epoch": epoch + 1,
                "train_cl/loss": mean_loss,
                "train_cl/masked_loss": mean_masked_loss,
                "tran_cl/distill_loss": mean_distill_loss,
                "train_cl/buffer_loss": mean_buffer_loss,
                "train_cl/align_loss": mean_align_loss,
                "cl_lr": learning_rate,
            }, step = epoch + 1)
            
            print(f'Epoch {epoch+1}: Masked Loss: {mean_masked_loss:.6f}')
            print(f'Epoch {epoch+1}: Distillation Loss: {mean_distill_loss:.6f}')
            print(f'Epoch {epoch+1}: Buffer Loss: {mean_buffer_loss:.6f}')
            print(f'Epoch {epoch+1}: Alignment Loss: {mean_align_loss:.6f}')
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.6f}')
            # Choose the best metric depending on masking
            metric = mean_masked_loss if getattr(self.params, 'need_mask', False) else mean_loss
            if metric < best_loss:
                model_path = rf'{self.params.model_dir}/epoch{epoch+1}_loss{metric}.pth'
                torch.save(self.model.state_dict(), model_path)
                print("model save in " + model_path)
                best_loss = metric