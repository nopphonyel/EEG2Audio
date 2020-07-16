from main_code.dataset.numerical_stimuli.dataset import IMG_NUM_DATASET
from torchvision import transforms
import matplotlib.pyplot as plt

dataset = IMG_NUM_DATASET()
for i in range(10):
    img, label = dataset.test_get_item(i)
    final_img = transforms.ToPILImage()(img)
    plt.imshow(final_img, cmap='gray')
    plt.show()
