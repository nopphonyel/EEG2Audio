from main_code.dataset.my_eeg.eeg_numeric_dataset import EEG_NUM_STIM

dataset = EEG_NUM_STIM()
print(dataset.test_get_item(1)[1])
