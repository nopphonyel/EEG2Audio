import torch

from torch.utils.data import DataLoader
from main_code.dataset.my_eeg.eeg_numeric_dataset import EEG_NUM_STIM
from main_code.dataset.numerical_stimuli.dataset import IMG_NUM_DATASET

from main_code.model.img_classifier import VGG16_IMG_CLASSIFIER
from main_code.model.eeg_classifier import EEG_CLASSIFIER
from main_code.model.generator import GENERATOR_RGB
from main_code.model.discrim import DISCRIM_RGB

from tqdm import trange

# MODEL CONFIGURATION
NOISE_DIM_SIZE = 100
EEG_LATENT_FEATURES = NOISE_DIM_SIZE

EEG_CHANNEL = 8
CLASS_NUMBER = 10

img_classifier_model = VGG16_IMG_CLASSIFIER()  # << Expected pre-trained model.
img_classifier_model.load_state_dict(torch.load('model/exported/IMG_Classifier.pth'))

eeg_classifier = EEG_CLASSIFIER(eegChanNum=EEG_CHANNEL, outputSize=CLASS_NUMBER)  # << Expected pre-trained model.
eeg_classifier.load_state_dict(torch.load('model/exported/EEG_Classifier.pth'))

generator = GENERATOR_RGB(noise_dim=NOISE_DIM_SIZE, features_dim=EEG_LATENT_FEATURES)
discrim = DISCRIM_RGB(img_classifier=img_classifier_model)

# TRAIN CONFIGURATION
GEN_ADAM_LR = 0.00003
GEN_ADAM_BETA_1 = 0.5

DIS_ADAM_LR = 0.00005
DIS_ADAM_BETA_1 = 0.5

gen_optim = torch.optim.Adam(generator.parameters(), lr=GEN_ADAM_LR, betas=(GEN_ADAM_BETA_1, 0.999))
dis_optim = torch.optim.Adam(discrim.parameters(), lr=DIS_ADAM_LR, betas=(DIS_ADAM_BETA_1, 0.999))

EPOCH = 10000

# DATASET CONFIGURATION
BATCH_SIZE = 12
num_img_storage = IMG_NUM_DATASET()
eeg_dataset = EEG_NUM_STIM()
eeg_dataset_loader = DataLoader(dataset=eeg_dataset, batch_size=BATCH_SIZE, shuffle=True)

for each_epoch in trange(EPOCH):
    for eeg_data, each_label in eeg_dataset_loader:
        print(eeg_data.shape)
        # eeg_data = eeg_data.unsqueeze(1)
        eeg_data = eeg_data[:, :, :, -33:-1]
        eeg_features, _ = eeg_classifier(eeg_data)

        gen_img = generator(eeg_features)  # No need to gen noise, it's already generated in the model
        # Get the image tensor
        real_img = num_img_storage.get_raw_img_tensor(each_label.argmax(1))

        # Let fake = 1
        fake_label = torch.ones([12, 1])
        pred_is_fake, pred_img_class = discrim(gen_img)

        # Let real = 0
        real_label = torch.zeros([12, 1])
        pred_is_fake, pred_img_class = discrim(real_img)
