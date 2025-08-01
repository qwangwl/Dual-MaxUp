##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/aliasvishnu/EEGNet
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Feature Extractor """
from torch import nn
import torch.nn.functional as F
import torch

from .MLP import MLP_encoder
from .CNN import CNN_encoder
from .Transformer import Transformer_encoder
from .Dual_Attention import Dual_Attention_encoder

class Deap_FeatureExtractor(nn.Module):

    def __init__(self, model_type, n_channels=32, d_model=128, ratio=2, dropout=0.5):
        super(Deap_FeatureExtractor, self).__init__()
        if model_type == 'MLP':
            self.encoder = MLP_encoder(n_channels=n_channels, d_model=d_model, ratio=ratio, drop_prob=dropout)
        elif model_type == 'CNN':
            self.encoder = CNN_encoder(n_channels=n_channels, d_model=d_model, ratio=ratio, drop_prob=dropout)
        elif model_type == 'Transformer':
            self.encoder = Transformer_encoder(n_channels=n_channels, d_model=d_model, ratio=ratio, drop_prob=dropout)
        elif model_type == 'Dual-Attention':
            self.encoder = Dual_Attention_encoder(n_channels=n_channels, d_model=d_model, ratio=ratio, drop_prob=dropout)

        self.out_features = self.encoder.out_features

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        return x

