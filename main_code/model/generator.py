import torch
from torch import nn
from main_code.model.layer.MoGLayer import MoGLayer
from main_code.model.discrim import DISCRIM_RGB


def gen_noise(noise_dim):
    return torch.rand(noise_dim)


class GENERATOR_RGB(nn.Module):
    def __init__(self, noise_dim, features_dim):
        super(GENERATOR_RGB, self).__init__()
        assert features_dim == noise_dim, "<X>: Currently, noise and features dimension expect to be same."
        self.noise_dim = noise_dim

        # From equation z = MEANi + (STDi * EP) | EP ~ N(0,1)
        self.MoGLayer = MoGLayer(noise_dim=noise_dim)

        self.dense01 = nn.Sequential(
            nn.Linear(in_features=features_dim, out_features=features_dim)
            , nn.Tanh()
        )

        self.batchNorm01 = nn.BatchNorm1d(num_features=features_dim)

        self.dense02 = nn.Sequential(
            nn.Linear(in_features=features_dim, out_features=512 * 4 * 4)
            , nn.ReLU()
        )

        self.conv2dT01 = nn.Sequential(
            # From the paper, they use kern_size = 5, padding = 0
            nn.BatchNorm2d(num_features=512, momentum=0.8)
            , nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, padding=1, stride=2)
            , nn.ReLU()
        )

        self.conv2dT02 = nn.Sequential(
            nn.BatchNorm2d(num_features=256, momentum=0.8)
            , nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2)
            , nn.ReLU()
        )

        self.conv2dT03 = nn.Sequential(
            nn.BatchNorm2d(num_features=128, momentum=0.8)
            , nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
            , nn.ReLU()
        )

        self.conv2dT04 = nn.Sequential(
            nn.BatchNorm2d(num_features=64, momentum=0.8)
            , nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, padding=1, stride=2)
            , nn.ReLU()
        )

        self.tanh = nn.Tanh()

    """
    Expected tensor to have shape (N,C,...)
    """
    def forward(self, eeg_features):
        noise_input = gen_noise(self.noise_dim)
        x = self.MoGLayer(noise_input)
        x = self.dense01(x)
        x = x * eeg_features  # Multiply noise with eeg signal here
        print(x.shape)  # Expected to be 2 dimension

        # if len(x.shape) >= 2:
        x = self.batchNorm01(x)  # This layer allowed one batch when the model in eval mode.
        x = self.dense02(x)

        x = x.reshape([x.shape[0], 512, 4, 4])
        # x = x.unsqueeze(0) # Try to run the code with out this line first
        x = self.conv2dT01(x)
        x = self.conv2dT02(x)
        x = self.conv2dT03(x)
        x = self.conv2dT04(x)
        x = self.tanh(x)
        return x
