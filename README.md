# EEG2Audio
This project is in feasibility study... I'm not even think that it will be fully working

#####Possible issues
- Where do we need to place the electrode?
  - Currently place around the temporal frontal lobe...
    - Reference?

#####Project details
- This project will be using the conditional GANs since we need to input 
the EEG signal to generate the spectogram of audio
  - We may use the WGAN to improve the traning stability
  - The discriminator and generator model will use the same arch from EEG2Image paper
    - Then we tweaks the model later.