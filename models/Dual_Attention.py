import torch
import torch.nn as nn
import torch.nn.functional as F

class FABlock(nn.Module):
    def __init__(self, d_model, ratio=2):
        super(FABlock, self).__init__()

        self.cGAP = nn.AdaptiveAvgPool1d(1)
        self.f_linear = nn.Conv1d(in_channels=d_model, out_channels=d_model // ratio, kernel_size=1)
        self.s_linear = nn.Conv1d(in_channels=d_model // ratio, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        # x shape (B, N, C)
        f = self.cGAP(x)

        f = self.f_linear(f)
        f = F.relu(f)
        f = self.s_linear(f)

        return torch.sigmoid(f) * x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # '-1' means last dimension.
        var = x.var(-1, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta

        return out

class Dual_Attention_encoder(nn.Module):
    def __init__(self, n_channels, d_model, ratio, drop_prob=0.1):
        super(Dual_Attention_encoder, self).__init__()
        self.caBlock = FABlock(n_channels, ratio)
        self.faBlock = FABlock(d_model, ratio)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        # self.conv1 = nn.Conv1d(in_channels=128, out_channels=60, kernel_size=1)
        self.out_features = n_channels * d_model

    def forward(self, x, mask=None):

        x = self.caBlock(x)
        x = self.ln1(self.dropout1(x))

        x = x.permute(0, 2, 1)
        x = self.faBlock(x)
        x = x.permute(0, 2, 1)

        x = self.ln2(self.dropout2(x))
        # print(x.shape)
        # x = x.permute(0, 2, 1)
        # x = self.conv1(x)
        return x


if __name__ == "__main__":

    x = torch.randn(size = (10, 32, 128))

    net = Dual_Attention_encoder(n_channels=32, d_model=128, ratio=1)

    print(net(x).shape)
    print(sum(param.numel() for param in net.parameters()))