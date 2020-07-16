from torch import nn
import torch


class EEG_CLASSIFIER(nn.Module):
    def __init__(self, eegChanNum=14, outputSize=10):
        super(EEG_CLASSIFIER, self).__init__()
        ## (BATCH, CHAN, EEG_CHAN, EEG_LEN)
        self.batchNorm_1 = nn.BatchNorm2d(num_features=1)
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 4))
            , nn.ReLU()
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=25, kernel_size=(eegChanNum, 1))
            , nn.ReLU()
        )

        self.maxPool_1 = nn.MaxPool2d(kernel_size=(1, 3))

        self.conv2d_3 = nn.Sequential(  # Todo: Recheck the data format
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(4, 25))
            , nn.ReLU()
        )

        self.maxPool_2 = nn.MaxPool2d(kernel_size=(1, 3))

        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(50, 2))
            , nn.ReLU()
        )

        self.flatten_1 = self.flatten

        self.batchNorm_2 = nn.BatchNorm1d(num_features=100)

        self.final_dense_1 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100)
            , nn.ReLU()
        )

        self.batchNorm_3 = nn.BatchNorm1d(num_features=100)

        self.final_dense_2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=outputSize)
            , nn.Softmax(dim=1)
        )

    def flatten(self, tensor):
        total = 1
        for i in range(1, len(tensor.shape)):
            total = total * tensor.shape[i]
        return tensor.reshape([tensor.shape[0], total])

    def channel_first(self, tensor):
        assert len(tensor.shape) == 4, "Expect 4D tensor input."
        x = torch.transpose(tensor, 1, 2)
        return torch.transpose(x, 2, 3)

    def dechannel_first(self, tensor):
        assert len(tensor.shape) == 4, "Expect 4D tensor input."
        x = torch.transpose(tensor, 2, 3)
        return torch.transpose(x, 1, 2)

    """
    Expected tensor to have shape (N,C,H,W)
    N : Batch size
    C : Channel
    H : Electrode or EEG Channel (Default is 14)
    W : Sample len (The ThoughtViz use len = 32)
    """

    def forward(self, eeg):
        x = self.batchNorm_1(eeg)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.maxPool_1(x)

        x = self.channel_first(x)  # This method just swap channel with H and W
        x = self.conv2d_3(x)
        x = self.dechannel_first(x)
        x = self.maxPool_2(x)
        x = self.conv2d_4(x)
        x = self.flatten(x)
        x = self.batchNorm_2(x)
        eeg_features = self.final_dense_1(x)
        x = self.batchNorm_3(eeg_features)
        eeg_class = self.final_dense_2(x)
        return eeg_features, eeg_class
