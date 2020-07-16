import pandas as pd
import pickle
from torch.tensor import Tensor
import torch

EEG_SAMPLE_PATH = 'eeg.csv'

eeg_df = pd.read_csv(EEG_SAMPLE_PATH)
idx_list = eeg_df[eeg_df['Marker'] != '0'].index.tolist()
datazone_dict = {}
for i in range(len(idx_list)):
    idx = idx_list[i]
    marker = eeg_df.iloc[idx]['Marker']
    if i + 1 == len(idx_list):
        idx_next = eeg_df.tail(1).index.item()
        datazone_dict[marker] = (idx, idx_next)
        # print("Marker", marker, "From", idx, "to", idx_next)
        break
    else:
        idx_next = idx_list[i + 1]
        datazone_dict[marker] = (idx, idx_next)
        # print("Marker", marker, "From", idx, "to", idx_next)

eeg_stim_dataset = []
eeg_think_dataset = []
min_len_stim = 10000
min_len_think = min_len_stim
for each_key in datazone_dict:
    data_range = datazone_dict[each_key]
    marker = each_key.split(",")[1:4]

    eeg = eeg_df.iloc[data_range[0]:data_range[1]]
    del eeg['timestamps']
    del eeg['Marker']

    eeg_numpy = eeg.to_numpy(dtype='float')
    eeg_tensor = Tensor(eeg_numpy)
    marker_1hot = torch.zeros(10)

    if marker[1] == '*' and marker[2] == '*':
        # Stimuli case
        marker_1hot[int(marker[0])] = 1
        if min_len_stim > eeg_tensor.shape[0]:
            min_len_stim = eeg_tensor.shape[0]

        # Reformat the tensor
        eeg_tensor = eeg_tensor[-751:-1, :]
        eeg_tensor = torch.transpose(eeg_tensor,0,1).unsqueeze(0)

        eeg_tuple_stim = (eeg_tensor, marker_1hot)
        #print(eeg_tuple_stim[0].shape)
        print("STIM",marker[0])

        eeg_stim_dataset.append(eeg_tuple_stim)
    elif marker[0] == '*' and marker[2] == '*':
        # Think case
        marker_1hot[int(marker[1])] = 1
        if min_len_think > eeg_tensor.shape[0]:
            min_len_think = eeg_tensor.shape[0]

        # Reformat the tensor
        eeg_tensor = eeg_tensor[-1400:-1, :]
        eeg_tensor = torch.transpose(eeg_tensor,0,1).unsqueeze(0)

        eeg_tuple_think = (eeg_tensor, marker_1hot)
        #print(eeg_tuple_think[0].shape)
        eeg_think_dataset.append(eeg_tuple_think)
    else:
        # Do nothing on fixation case
        pass

    # print(eeg_stim_dataset)

'''Select last 7th element for test set'''
print(min_len_stim, min_len_think)
pickle.dump(eeg_stim_dataset[0:23], open("eeg_stim_train.dat", "wb"))
pickle.dump(eeg_think_dataset[0:23], open("eeg_think_train.dat", "wb"))
pickle.dump(eeg_stim_dataset[23:30], open("eeg_stim_test.dat", "wb"))
pickle.dump(eeg_think_dataset[23:30], open("eeg_think_test.dat", "wb"))
