from main_code.dataset.my_eeg.eeg_numeric_dataset import EEG_NUM_STIM
from main_code.model.eeg_classifier import EEG_CLASSIFIER
from torch.utils.data import DataLoader
import torch
from tqdm import trange

import matplotlib.pyplot as plt
import numpy as np

dataset_train = EEG_NUM_STIM()
dataset_eval = EEG_NUM_STIM()
eeg, label = dataset_train.test_get_item(2)
cut_eeg = eeg[-33:-1, :]


# print(eeg.shape, cut_eeg.shape)
# print(dataset.test_len())

def logging_plot(train_hist: list, val_hist: list):
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Val Loss')
    plt.legend()
    plt.show()


# Model Configuration
EEG_CHANNEL = 8
model = EEG_CLASSIFIER(eegChanNum=EEG_CHANNEL)

# Training configuration
DEV = "cuda"
EPCH = 100000
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_EVAL = 1
EXPORT_PATH = 'model/exported/'
EXPORT_NAME = 'EEG_Classifier.pth'
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
loader_eval = DataLoader(dataset_eval, batch_size=BATCH_SIZE_EVAL, shuffle=True)

# Optimizer configuration
LR = 0.0001
DECAY = 1e-6  # Currently not sure what decay in original implementation is...
MOMENTUM = 0.9
NESTEROV = True
# Definitely not LR decay since we don't know exact number of epoch
#optim = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=DECAY, momentum=MOMENTUM, nesterov=NESTEROV)
optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)

# Loss configuration
loss_func = torch.nn.CrossEntropyLoss()

model.to(DEV)
train_hist = []
val_hist = []
min_val_loss = None
for epch_id in trange(EPCH):
    # Training phase
    model.train()
    each_train_hist = []
    for each_eeg, each_label in loader_train:
        optim.zero_grad()
        # Move data to DEV
        each_eeg = each_eeg[:, :, :, -33:-1].to(DEV)
        each_label = each_label.to(DEV)

        predict_features, predict_class = model(each_eeg)
        each_label = each_label.argmax(1)

        loss_val = loss_func(predict_class, each_label)
        each_train_hist.append(loss_val.item())
        loss_val.backward()
        optim.step()

    train_hist.append(np.average(each_train_hist))

    # Eval phase
    model.eval()
    each_val_hist = []
    for each_eeg, each_label in loader_eval:
        # Move data to DEV
        each_eeg = each_eeg[:, :, :, -33:-1].to(DEV)
        each_label = each_label.to(DEV)

        predict_features, predict_class = model(each_eeg)
        each_label = each_label.argmax(1)

        loss_val = loss_func(predict_class, each_label)
        each_val_hist.append(loss_val.item())

    avg_val_hist = np.average(each_val_hist)
    val_hist.append(avg_val_hist)

    if min_val_loss is None:
        min_val_loss = avg_val_hist
    else:
        if avg_val_hist < min_val_loss:
            min_val_loss = avg_val_hist
            torch.save(model.state_dict(), EXPORT_PATH+EXPORT_NAME)

    if epch_id % 1000 == 0:
        logging_plot(train_hist, val_hist)
