import torch

blank_tensor = torch.zeros((3,1))
print(blank_tensor)

x = torch.nn.init.xavier_uniform_(blank_tensor)
print(x, blank_tensor)

y = torch.nn.init.zeros_(blank_tensor)
print(y, blank_tensor)



