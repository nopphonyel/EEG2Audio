import torch
from main_code.model.eeg_classifier import EEG_CLASSIFIER
from main_code.model.discrim import DISCRIM_RGB
from main_code.model.generator import GENERATOR_RGB
from main_code.model.img_classifier import VGG16_IMG_CLASSIFIER

SIGNAL_LEN = 32
FEATURES = 14  # (Electrodes)
CLASS_NUM = 10

test_eeg_classifier = EEG_CLASSIFIER(eegChanNum=FEATURES, outputSize=CLASS_NUM)
test_img_classifier = VGG16_IMG_CLASSIFIER()
test_discrim = DISCRIM_RGB(None)
test_generator = GENERATOR_RGB(10, 10)

# eeg_sample = torch.rand([44367, 1, FEATURES, SIGNAL_LEN])
# test_eeg_classifier(eeg_sample)

CHANNEL = 3
HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 16

img_sample = torch.rand([BATCH_SIZE, CHANNEL, HEIGHT, WIDTH])
test_img_classifier(img_sample)

# test_model(eeg_sample)
# fake_or_real = test_discrim(img_sample)

# img_sample = torch.rand([BATCH_SIZE, CHANNEL, HEIGHT, WIDTH])
# output = test_discrim(img_sample)

# eeg_features_sample = torch.rand([1, 10])
# test_generator.eval()
# output = test_generator(eeg_features_sample)
