import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod
from .biot import BIOTEncoder

class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.chan_conv = Conv1dWithConstraint(16, 19, 1, max_norm=1.0)
        self.biot = BIOTEncoder(n_channels=19)
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            # print(torch.load(param.foundation_dir, map_location=map_location).keys())
            # print(torch.load(param.foundation_dir, map_location=map_location)['state_dict'].keys())
            # print(self.biot.state_dict().keys())
            try:
                biot_state_dict = {k.replace('model.biot.', ''): v for k, v in torch.load(param.foundation_dir, map_location=map_location)['state_dict'].items() if k.startswith('model.biot.')}
            except:
                biot_state_dict = torch.load(param.foundation_dir, map_location=map_location)
                
            # print('biot_state_dict:', biot_state_dict.keys())
            self.biot.load_state_dict(biot_state_dict)

        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x = x / 100
        # print('Input shape to BIOT:', x.shape)
        # print(x.device)
        # bz, ch_num, seq_len, patch_size = x.shape
        feats = self.chan_conv(x.view(x.shape[0], x.shape[1], -1))
        # print('Shape after channel conv:', feats.shape)
        feats = self.biot(feats)
        out = self.classifier(feats)
        return out.squeeze(-1)
