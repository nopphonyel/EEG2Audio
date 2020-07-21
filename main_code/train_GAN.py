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
DEV = "cuda:1"
NOISE_DIM_SIZE = 100
EEG_LATENT_FEATURES = NOISE_DIM_SIZE

EEG_CHANNEL = 8
CLASS_NUMBER = 10

img_classifier_model = VGG16_IMG_CLASSIFIER()  # << Expected pre-trained model.
img_classifier_model.load_state_dict(torch.load('model/exported/IMG_Classifier.pth'))
img_classifier_model.to(DEV)

eeg_classifier = EEG_CLASSIFIER(eegChanNum=EEG_CHANNEL, outputSize=CLASS_NUMBER)  # << Expected pre-trained model.
eeg_classifier.load_state_dict(torch.load('model/exported/EEG_Classifier.pth'))
eeg_classifier.to(DEV)

generator = GENERATOR_RGB(noise_dim=NOISE_DIM_SIZE, features_dim=EEG_LATENT_FEATURES).to(DEV)
generator.set_dev(DEV)
generator.train()
discrim = DISCRIM_RGB(img_classifier=img_classifier_model).to(DEV)
discrim.train()

# TRAIN CONFIGURATION
EPOCH = 10000

GEN_ADAM_LR = 0.00003
GEN_ADAM_BETA_1 = 0.5

DIS_ADAM_LR = 0.00005
DIS_ADAM_BETA_1 = 0.5

gen_optim = torch.optim.Adam(generator.parameters(), lr=GEN_ADAM_LR, betas=(GEN_ADAM_BETA_1, 0.999))
dis_optim = torch.optim.Adam(discrim.parameters(), lr=DIS_ADAM_LR, betas=(DIS_ADAM_BETA_1, 0.999))

loss_func = torch.nn.BCELoss()

# DATASET CONFIGURATION
BATCH_SIZE = 12
num_img_storage = IMG_NUM_DATASET()
eeg_dataset = EEG_NUM_STIM()
eeg_dataset_loader = DataLoader(dataset=eeg_dataset, batch_size=BATCH_SIZE, shuffle=True)


def train_discrim(label, features):
    gen_img = generator(features)
    # Get the image tensor
    real_img = num_img_storage.get_raw_img_tensor(label.argmax(1)).to(DEV)

    # Train Discriminator
    discrim.train()
    dis_optim.zero_grad()

    # Let real = 1
    real_label = torch.ones([label.shape[0], 1]).to(DEV)
    predict, _ = discrim(real_img)
    d_loss_real = loss_func(predict, real_label)  # << Replace with img real loss calculation

    # Let fake = 0 (train the fake img)
    fake_label = torch.zeros([label.shape[0], 1]).to(DEV)
    predict, _ = discrim(gen_img)
    d_loss_fake = loss_func(predict, fake_label)  # << Replace with img fake loss calculation

    d_loss = d_loss_fake + d_loss_real
    d_loss.backward()
    dis_optim.step()
    # End train discriminator


def train_generator(label, features):
    # Train generator
    discrim.eval()
    gen_optim.zero_grad()
    gen_img = generator(features)

    real_label = torch.ones([label.shape[0], 1]).to(DEV)

    # real_label = torch.ones([each_label.shape[0], 1]).to(DEV)
    predict, _ = discrim(gen_img)
    g_loss = loss_func(predict, real_label)  # Hope that generator can fool the discriminator
    g_loss.backward()
    gen_optim.step()


for each_epoch in trange(EPOCH):
    for eeg_data, each_label in eeg_dataset_loader:
        # print(eeg_data.shape)
        # eeg_data = eeg_data.unsqueeze(1)
        eeg_data = eeg_data[:, :, :, -33:-1].to(DEV)
        eeg_features, _ = eeg_classifier(eeg_data)
        eeg_features = eeg_features.to(DEV)

        train_discrim(each_label, eeg_features)
        train_generator(each_label, eeg_features)
