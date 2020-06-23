import torch
from torch.utils.data import TensorDataset


class DummyDataset(TensorDataset):
    __DATA_SET_SIZE = 1500

    __EEG_LEN = 32

    __IMG_H = 64
    __IMG_W = 64

    __IMG_CLASS_NUM = 10

    def __init__(self):
        super(DummyDataset, self).__init__()

        # [EEG:14xLEN, IMG:3xHxW, IMG_CLASS:1x10]
        self.data_set = []

        # Pos 0 for EEG
        self.data_set.append(torch.rand([self.__DATA_SET_SIZE, 14, self.__EEG_LEN]))
        # Pos 1 for IMG
        self.data_set.append(torch.rand([self.__DATA_SET_SIZE, 3, self.__IMG_H, self.__IMG_W]))
        # Pos 2 for IMG_Class
        self.data_set.append(torch.rand([self.__DATA_SET_SIZE, self.__IMG_CLASS_NUM]))

    def __getitem__(self, idx):
        return self.data_set[0][idx], self.data_set[1][idx], self.data_set[2][idx]

    def __len__(self):
        return self.__DATA_SET_SIZE

    def test_get_item(self, idx):
        return self.__getitem__(idx)
