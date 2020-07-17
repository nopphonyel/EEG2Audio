from PIL import Image
import random
from torchvision import transforms
import glob
import torch
from torch.utils.data import TensorDataset


class IMG_NUM_DATASET(TensorDataset):
    # Every time that get the data, it will be generated the noise + IMG
    def __init__(self):
        super(IMG_NUM_DATASET, self).__init__()
        self.img_num = [None] * 10
        self.label = [None] * 10
        # Max noise opacity
        self.MAX_OPACITY = 75
        self.__load_img()

    def __load_img(self):
        self.img_list = glob.glob("dataset/numerical_stimuli/png/*.png")
        for eimg_path in self.img_list:
            img = Image.open(eimg_path, "r")
            # Get the right index for store the image
            idx = int(img.filename[-5:-4])
            # Transform image to tensor
            img_tensor = transforms.ToTensor()(img)
            self.img_num[idx] = img_tensor
            # Transform label to one hot encoded tensor
            label_tensor = torch.tensor([0] * 10, dtype=torch.float)
            label_tensor[idx] = 1
            self.label[idx] = label_tensor

    def __len__(self):
        return 128

    def __add_noise(self, orig_img):
        rand_opacity = (random.randint(0, self.MAX_OPACITY)) / 100
        noise_tensor = torch.rand([1, 64, 64])
        final_tensor = (orig_img * (1 - rand_opacity)) + (noise_tensor * rand_opacity)
        return final_tensor

    def __getitem__(self, idx):
        img_num = self.img_num[idx % 10]
        return self.__add_noise(img_num), self.label[idx % 10]

    def get_raw_img_tensor(self, class_id: torch.Tensor):  # Expected to be a tensor of integer (Not 1hot encoded)
        pack = None
        for i in class_id:
            if pack is None:
                pack = self.img_num[i.item()].unsqueeze(0)
            else:
                pack = torch.cat((pack, self.img_num[i.item()].unsqueeze(0)), 0)

        return pack

    def test_get_item(self, idx):
        return self.__getitem__(idx)


def hidden():
    img_list = glob.glob("png/*.png")

    img = Image.open(r"png/6.png")

    INTENSITY = 0.8
    img_tensor = transforms.ToTensor()(img)
    noise_tensor = torch.rand([1, 64, 64])
    final_tensor = (img_tensor * (1 - INTENSITY)) + (noise_tensor * INTENSITY)

    final_img = transforms.ToPILImage()(final_tensor)
    final_img.save("test.jpg")
