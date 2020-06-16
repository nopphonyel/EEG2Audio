import torch
from torch import nn
from main_code.model.layer.MoGLayer import MoGLayer


def __gen_noise(noise_dim):
    return torch.rand(noise_dim)


class GENERATOR_RGB(nn.Module):
    def __init__(self, noise_dim, features_dim):
        super(GENERATOR_RGB, self).__init__()
        self.noise_dim = noise_dim

        # From equation z = MEANi + (STDi * EP) | EP ~ N(0,1)
        self.MoGLayer = MoGLayer(noise_dim=noise_dim)
        self.dense01 = nn.Sequential(
            nn.Linear(in_features=features_dim, out_features=features_dim)
            , nn.Tanh()
        )
        self.dense02 = nn.Sequential(
            nn.Linear(in_features=features_dim, out_features=512 * 4 * 4)
            , nn.ReLU()
        )

        self.conv2dT01 = nn.Sequential(
            nn.BatchNorm2d(num_features=1)
            , nn.ConvTranspose2d(in_channels=1,out_channels=256, kernel_size=5, stride=2)
            , nn.ReLU
        )

    def forward(self, eeg_features):
        noise_input = self.gen_noise(self.noise_dim)
        x = self.MoGLayer(eeg_features, noise_input)
        x = self.dense01(x)
        print(x.shape)
