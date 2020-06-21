from torch import nn
import torch


class EEG_CLASSIFIER(nn.Module):
    def __init__(self, eegChanNum, outputClassNum):
        super(EEG_CLASSIFIER, self).__init__()
        self.batchNorm_1st = nn.BatchNorm1d(num_features=eegChanNum)
        self.conv1d32 = nn.Sequential(
            nn.Conv1d(in_channels=eegChanNum, out_channels=32, kernel_size=4)
            , nn.ReLU()
        )
        self.conv1d25 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=25, kernel_size=eegChanNum)
            , nn.ReLU()
        )

        self.conv2d50 = nn.Sequential(  # Data format channel_first Todo: recheck the data format here
            nn.Conv1d(in_channels=25, out_channels=50, kernel_size=(25, 4))
            , nn.ReLU()
        )

        self.conv2d100 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=100, kernel_size=(2, 50))
            , nn.ReLU()
        )
        # 99 Here is test number
        self.dense01 = nn.Sequential(
            nn.Linear(in_features=7245000, out_features=100)
            , nn.ReLU()
        )

        self.dense02 = nn.Sequential(
            nn.Linear(in_features=100, out_features=outputClassNum)
            , nn.Softmax()
        )

        ### Please continue here
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.maxPool = nn.MaxPool2d(kernel_size=(1, 3))

        self.batchNorm_2nd = nn.InstanceNorm1d(num_features=90)  # Todo: recheck
        self.batchNorm100 = nn.BatchNorm2d(num_features=100)

    def __checkShape(self, x):
        if len(x.shape) == 2:
            return x.unsqueeze(0)

    """
    Expected tensor to have shape (N,C,O,???,...)
    """
    def forward(self, eeg):
        eeg = eeg.transpose(0, 1)  # Swap dimension between dimension 1 and 0
        x = self.batchNorm_1st(eeg)  # due to batch norm got something weird
        x = x.transpose(0, 1)  # switch back
        x = self.__checkShape(x)  # Add another dimension to tensor

        x = self.conv1d32(x)
        x = self.conv1d25(x)
        x = self.maxPool(x)

        # Since 'self.conv2d50' output the format as (batch_size, channels, height, width)
        # I need to unsqueeze() the 2nd (index = 1) dim
        x = x.unsqueeze(3)
        x = self.conv2d50(x)  # How this even possible? from 1D convolution to 2D convolution????
        x = self.maxPool(x)
        x = self.conv2d100(x)
        x = torch.flatten(x)
        # x = self.batchNorm_2nd(x)
        eeg_features = self.dense01(x)
        eeg_class = self.batchNorm100(eeg_features)
        eeg_class = self.dense02(eeg_class)
        return eeg_features, eeg_class
