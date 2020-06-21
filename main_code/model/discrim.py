from prompt_toolkit import output
from torch import nn
import torch
import torch.nn.functional as func


class DISCRIM_RGB(nn.Module):
    """
    Currently, this model require a batch size
    """

    def __init__(self, img_classifier):
        super(DISCRIM_RGB, self).__init__()

        self.img_classifier = img_classifier

        self.conv2dPack = nn.ModuleList()
        self.conv2dPack.append(
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=16)
                , nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=32)
                , nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=64)
                , nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1), nn.LeakyReLU(0.2)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=128)
                , nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=256)
                , nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )

        # self.flatten = nn.Sequential(
        #    nn.Linear(in_features=''' Do dimension calculation here ''', out_features=1),
        #    nn.Sigmoid()
        # )

        self.sigmoid = nn.Sigmoid()

    def __find_total_dim(self, tensor):
        total = 1
        for i in range(1, len(tensor.shape)):
            total = total * tensor.shape[i]
        return total

    """
    Expected tensor to have shape (N,C,H,W,...)
    """
    def forward(self, img):
        for idx, conv2d_layer in enumerate(self.conv2dPack):
            print("LAYER :", idx)
            img = conv2d_layer(img)
            print(img.shape)

        flatten_shape = self.__find_total_dim(img)
        x = img.reshape([img.shape[0], flatten_shape])  # flatten operation
        self.dense = nn.Linear(in_features=x.shape[1], out_features=1)  # Todo: recheck the input_features.

        x = self.dense(x)
        fake = self.sigmoid(x)
        # img_class = self.img_classifier(x)
        return fake  # , img_class  # Todo: merge into a single tensor (Do we really need to merge?)
