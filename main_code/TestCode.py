from main_code.dataset.dummy_dataset import DummyDataset

dataset = DummyDataset()
sample_eeg, sample_label, sample_class = dataset.test_get_item(15)
print(sample_eeg.shape, sample_label.shape, sample_class.shape)
