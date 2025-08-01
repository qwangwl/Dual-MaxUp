import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_encoder(nn.Module):
    def __init__(self, n_channels, d_model, ratio, drop_prob=0.1):
        super(CNN_encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,kernel_size=(1, 16)),  #(1,32,128)->(16,32,113)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  #(16,32,113)->(16,32,56)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3)),  # (16,32,56)->(32,32,54)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # (32,32,54)->(32,32,27)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3)),  # (32,32,27)->(64,32,25)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # (64,32,25)->(64,32,12)

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(32, 1)),  # (64,32,12)->(32,1,12)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.out_features = 32*12


    def forward(self, x):
        x = x.unsqueeze(1)
        #print(x.shape)
        x = self.encoder(x)

        return x

if __name__ == "__main__":

    x = torch.randn(size = (10, 32, 128))
    net = CNN_encoder(n_channels=32, d_model=128, ratio=1)
    print(net(x).shape)
    print(sum(param.numel() for param in net.parameters()))