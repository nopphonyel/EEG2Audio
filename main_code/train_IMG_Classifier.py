from main_code.dataset.numerical_stimuli.dataset import IMG_NUM_DATASET
from main_code.model.img_classifier import VGG16_IMG_CLASSIFIER
from torch.utils.data import DataLoader
import torch
from tqdm import trange

import matplotlib.pyplot as plt
import numpy as np

dataset_train = IMG_NUM_DATASET()
dataset_eval = IMG_NUM_DATASET()


# print(eeg.shape, cut_eeg.shape)
# print(dataset.test_len())

def logging_plot(train_hist: list, val_hist: list):
    plt.plot(train_hist, label="Training loss")
    plt.plot(val_hist, label="Validation loss")
    plt.legend()
    plt.show()


# Model Configuration
EEG_CHANNEL = 8
model = VGG16_IMG_CLASSIFIER()

# Training configuration
DEV = "cuda"
EPCH = 10000
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 32
EXPORT_PATH = 'model/exported/'
EXPORT_NAME = 'IMG_Classifier.pth'
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
loader_eval = DataLoader(dataset_eval, batch_size=BATCH_SIZE_EVAL, shuffle=True)

# Optimizer configuration
LR = 5e-4
DECAY = 1e-3  # Currently not sure what decay in original implementation is...
MOMENTUM = 0.9
NESTEROV = True
optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)

# Loss configuration
loss_func = torch.nn.CrossEntropyLoss()

train_hist = []
val_hist = []

model.to(DEV)
for epch_id in trange(EPCH):
    # Training phase
    model.train()
    each_train_hist = []
    for each_img, each_label in loader_train:
        optim.zero_grad()
        # Move data to DEV
        each_img = each_img.to(DEV)
        each_label = each_label.to(DEV)

        predict_label = model(each_img)
        each_label = each_label.argmax(1)

        loss_val = loss_func(predict_label, each_label)
        each_train_hist.append(loss_val.item())
        loss_val.backward()
        optim.step()

    train_hist.append(np.average(each_train_hist))

    # Eval phase
    model.eval()
    each_val_hist = []
    for each_img, each_label in loader_eval:
        # Move data to DEV
        each_img = each_img.to(DEV)
        each_label = each_label.to(DEV)

        predict_label = model(each_img)
        each_label = each_label.argmax(1)

        loss_val = loss_func(predict_label, each_label)
        each_val_hist.append(loss_val.item())
    val_hist.append(np.average(each_val_hist))

    try:
        if val_hist[-1] < val_hist[-2]:
            torch.save(model.state_dict(), EXPORT_PATH+EXPORT_NAME)
    except IndexError:
        torch.save(model.state_dict(), EXPORT_PATH + EXPORT_NAME)

    if epch_id % 1000 == 0:
        logging_plot(train_hist, val_hist)
