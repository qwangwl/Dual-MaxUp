import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_encoder(nn.Module):
    def __init__(self, n_channels, d_model, ratio, drop_prob=0.1):
        super(MLP_encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_channels*d_model, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),
        )

        self.out_features = 128

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)

        return x



if __name__ == "__main__":

    """x = torch.randn(size = (10, 32, 128))

    net = MLP_encoder(n_channels=32, d_model=128, ratio=1)

    print(net(x).shape)
    print(sum(param.numel() for param in net.parameters()))"""

    classifier = BaseLearner(z_dim=4096)
    print(classifier.parameters()[0].size())
