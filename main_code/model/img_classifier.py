# For image classifier, we can use the implemented one from pytorch lib? VGG-16
import torch
from torch import nn


class VGG16_IMG_CLASSIFIER(nn.Module):
    def __init__(self):
        super(VGG16_IMG_CLASSIFIER, self).__init__()
        self.conv2d_1 = nn.Sequential(  # Todo: Padding same
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
            , nn.ReLU()
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
            , nn.ReLU()
        )

        self.maxpool_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            , nn.Dropout(0.25)
        )

        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
            , nn.ReLU()
        )

        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
            , nn.ReLU()
        )

        self.maxpool_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            , nn.Dropout(0.25)
        )

        self.conv2d_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
            , nn.ReLU()
        )

        self.conv2d_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
            , nn.ReLU()
        )

        self.maxpool_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            , nn.Dropout(0.25)
        )

        self.conv2d_7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
            , nn.ReLU()
        )

        self.conv2d_8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
            , nn.ReLU()
        )

        self.maxpool_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            , nn.Dropout(0.25)
        )

        self.dense_pack_1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512)
            , nn.ReLU()
            , nn.Dropout(0.25)
        )

        self.dense_pack_2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256)
            , nn.ReLU()
            , nn.Dropout(0.25)
        )

        self.dense_pack_3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128)
            , nn.ReLU()
            , nn.Dropout(0.25)
        )

        self.dense_pack_final = nn.Sequential(
            nn.Linear(in_features=128, out_features=10)
            , nn.Softmax(dim=1)
        )

    def flatten(self, tensor):
        total = 1
        for i in range(1, len(tensor.shape)):
            total = total * tensor.shape[i]
        return tensor.reshape([tensor.shape[0], total])

    def forward(self, img):
        x = self.conv2d_1(img)
        x = self.conv2d_2(x)
        x = self.maxpool_1(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.maxpool_2(x)
        x = self.conv2d_5(x)
        x = self.conv2d_6(x)
        x = self.maxpool_3(x)
        x = self.conv2d_7(x)
        x = self.conv2d_8(x)
        x = self.maxpool_4(x)

        x = self.flatten(x)
        x = self.dense_pack_1(x)
        x = self.dense_pack_2(x)
        x = self.dense_pack_3(x)
        x = self.dense_pack_final(x)

        return x
