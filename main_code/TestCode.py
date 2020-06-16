import torch
from main_code.model.eeg_classifier import EEG_CLASSIFIER

SIGNAL_LEN = 4860
FEATURES = 16  # (Electrodes)
OBSERVATION = 32  # Number of participants?

CLASS_NUM = 10

test_model = EEG_CLASSIFIER(eegChanNum=FEATURES, outputClassNum=CLASS_NUM)

# eeg_sample = torch.rand([15, FEATURES, OBSERVATION, SIGNAL_LEN])
eeg_sample = torch.rand([FEATURES, SIGNAL_LEN])
test_model(eeg_sample)
