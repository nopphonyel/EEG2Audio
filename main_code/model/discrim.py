from torch import nn


class DISCRIM_RGB(nn.Module):

    def __init__(self):
        super(DISCRIM_RGB, self).__init__()
        # Nonthing here

    def forward(self, img):
        x = nn.Conv2d(3, 3, stride=2)(img)
