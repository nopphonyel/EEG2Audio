import torch
import pickle
from torch.utils.data import TensorDataset

'''
Since there is not much data, I decided to store all of it in the class
'''


class EEG_NUM_STIM(TensorDataset):
    def __init__(self):
        super(EEG_NUM_STIM, self).__init__()
        self.train()

    def train(self):
        self.train_state = True
        self.data_set = pickle.load(open("dataset/my_eeg/eeg_stim_train.dat", "rb"))

    def eval(self):
        self.train_state = False
        self.data_set = pickle.load(open("dataset/my_eeg/eeg_stim_test.dat", "rb"))

    def __getitem__(self, idx):
        return self.data_set[idx][0], self.data_set[idx][1]

    def __len__(self):
        return len(self.data_set)

    def test_get_item(self, idx):
        return self.__getitem__(idx)

    def test_len(self):
        return self.__len__()


class EEG_NUM_THINK(EEG_NUM_STIM):
    def __init__(self):
        super(EEG_NUM_THINK, self).__init__()
        self.train()

    def train(self):
        self.train_state = True
        self.data_set = pickle.load(open("dataset/my_eeg/eeg_think_train.dat", "rb"))

    def eval(self):
        self.train_state = False
        self.data_set = pickle.load(open("dataset/my_eeg/eeg_think_test.dat", "rb"))
