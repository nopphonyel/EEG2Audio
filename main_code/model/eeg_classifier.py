from torch import nn
import torch

class EEG_CLASSIFIER(nn.Module):
    def __init__(self, eegChanNum, outputClassNum):
        super(EEG_CLASSIFIER, self).__init__()
        self.batchNorm_1st = nn.BatchNorm2d(num_features=eegChanNum)
        self.conv2d32 = nn.Sequential(
            nn.Conv2d(in_channels=eegChanNum, out_channels=32, kernel_size=(1, 4))
            , nn.ReLU()
        )
        self.conv2d25 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=25, kernel_size=(eegChanNum, 1))
            , nn.ReLU()
        )

        self.conv2d50 = nn.Sequential(  # Data format channel_first Todo: recheck the data format here
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(4, 25))
            , nn.ReLU()
        )

        self.conv2d100 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(50, 2))
            , nn.ReLU()
        )
        # 99 Here is test number
        self.dense01 = nn.Sequential(
            nn.Linear(in_features=99, out_features=100)
            , nn.ReLU()
        )

        self.dense02 = nn.Sequential(
            nn.Linear(in_features=100, out_features=outputClassNum)
            , nn.Softmax()
        )

        ### Please continue here
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.maxPool = nn.MaxPool2d((1, 3))

        self.batchNorm_2nd = nn.BatchNorm1d(num_features=1)  # Todo: recheck
        self.batchNorm100 = nn.BatchNorm2d(num_features=100)

    def forward(self, eeg):
        x = self.batchNorm_1st(eeg)
        x = self.conv2d32(x)
        x = self.conv2d25(x)
        x = self.maxPool(x)
        x = self.conv2d50(x)
        x = self.maxPool(x)
        x = self.conv2d100(x)
        x = torch.flatten(x)
        x = self.batchNorm_2nd(x)
        eeg_features = self.dense01(x)
        eeg_class = self.batchNorm100(eeg_features)
        eeg_class = self.dense02(eeg_class)
        return eeg_features, eeg_class
