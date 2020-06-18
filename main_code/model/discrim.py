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
        self.conv2dPack.append(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2))
        self.conv2dPack.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1))
        self.conv2dPack.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2))
        self.conv2dPack.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1))
        self.conv2dPack.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2))
        self.conv2dPack.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1))

        self.batchNormPack = nn.ModuleList()
        self.batchNormPack.append(nn.BatchNorm2d(num_features=16))
        self.batchNormPack.append(nn.BatchNorm2d(num_features=32))
        self.batchNormPack.append(nn.BatchNorm2d(num_features=64))
        self.batchNormPack.append(nn.BatchNorm2d(num_features=128))
        self.batchNormPack.append(nn.BatchNorm2d(num_features=256))

        self.batchNormTest = nn.BatchNorm1d(num_features=16)

        # self.flatten = nn.Sequential(
        #    nn.Linear(in_features=''' Do dimension calculation here ''', out_features=1),
        #    nn.Sigmoid()
        # )

        self.sigmoid = nn.Sigmoid()

        self.drop_things = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )

    def forward(self, img):
        for idx, (conv2d_layer, batchNorm_layer) in enumerate(zip(self.conv2dPack, self.batchNormPack)):
            print("LAYER :", idx)
            if idx == 0:
                x = conv2d_layer(img)
            else:
                x = conv2d_layer(x)
            x = self.drop_things(x)
            if idx == 5:  # Break before do batch_norm
                break
            x = batchNorm_layer(x)
            print(x.shape)

        print(x.shape)
        x = torch.flatten(x)  # flatten operation
        self.dense = nn.Linear(in_features=x.shape[0], out_features=1)  # Todo: recheck the input_features.

        x = self.dense(x)
        fake = self.sigmoid(x)
        #img_class = self.img_classifier(x)
        return fake#, img_class  # Todo: merge into a single tensor (Do we really need to merge?)
