import torch

from torch.utils.data import DataLoader
from main_code.dataset.dummy_dataset import DummyDataset

from main_code.model.img_classifier import VGG16_IMG_CLASSIFIER
from main_code.model.eeg_classifier import EEG_CLASSIFIER
from main_code.model.generator import GENERATOR_RGB
from main_code.model.discrim import DISCRIM_RGB

from tqdm import trange

# MODEL CONFIGURATION
NOISE_DIM_SIZE = 100
EEG_LATENT_FEATURES = NOISE_DIM_SIZE

EEG_CHANNEL = 14
CLASS_NUMBER = 10

img_classifier_model = VGG16_IMG_CLASSIFIER()  # << Expected pre-trained model.
eeg_classifier = EEG_CLASSIFIER(eegChanNum=EEG_CHANNEL, outputSize=CLASS_NUMBER) # << Expected pre-trained model.
generator = GENERATOR_RGB(noise_dim=NOISE_DIM_SIZE, features_dim=EEG_LATENT_FEATURES)
discrim = DISCRIM_RGB(img_classifier=img_classifier_model)

# TRAIN CONFIGURATION
GEN_ADAM_LR = 0.00003
GEN_ADAM_BETA_1 = 0.5

DIS_ADAM_LR = 0.00005
DIS_ADAM_BETA_1 = 0.5

gen_optim = torch.optim.Adam(generator.parameters(), lr=GEN_ADAM_LR, betas=(GEN_ADAM_BETA_1, 0.999))
dis_optim = torch.optim.Adam(discrim.parameters(), lr=DIS_ADAM_LR, betas=(DIS_ADAM_BETA_1, 0.999))

BATCH_SIZE = 48
EPOCH = 10000

# DATASET CONFIGURATION
dummy_img_dataset = DummyDataset()
dummy_dataset_loader = DataLoader(dataset=dummy_img_dataset, batch_size=BATCH_SIZE, shuffle=True)

for each_epoch in trange(EPOCH):
    for batch_id, (eeg_data, img_data, img_class_data) in enumerate(dummy_dataset_loader):
        print(eeg_data.shape)
        eeg_data = eeg_data.unsqueeze(1)
        eeg_features,_ = eeg_classifier(eeg_data)
        gen_img = generator(eeg_features)
        pred_is_fake, pred_img_class = discrim(gen_img)
