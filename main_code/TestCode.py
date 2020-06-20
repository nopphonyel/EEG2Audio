import torch
from main_code.model.eeg_classifier import EEG_CLASSIFIER
from main_code.model.discrim import DISCRIM_RGB
from main_code.model.generator import GENERATOR_RGB

SIGNAL_LEN = 4860
FEATURES = 16  # (Electrodes)
OBSERVATION = 32  # Number of participants?

CLASS_NUM = 10

test_model = EEG_CLASSIFIER(eegChanNum=FEATURES, outputClassNum=CLASS_NUM)
test_discrim = DISCRIM_RGB(None)
test_generator = GENERATOR_RGB(10, 10)

eeg_sample = torch.rand([15, FEATURES, OBSERVATION, SIGNAL_LEN])
eeg_sample = torch.rand([FEATURES, SIGNAL_LEN])

CHANNEL = 3
HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 16
img_sample = torch.rand([BATCH_SIZE, CHANNEL, HEIGHT, WIDTH])

# test_model(eeg_sample)
# fake_or_real = test_discrim(img_sample)

img_sample = torch.rand([BATCH_SIZE, CHANNEL, HEIGHT, WIDTH])
output = test_discrim(img_sample)

# eeg_features_sample = torch.rand([1, 10])
# test_generator.eval()
# output = test_generator(eeg_features_sample)
