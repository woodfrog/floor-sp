import _init_paths
import torch
import torch.nn as nn

from models import drn
import pdb


class CornerRoomAssociate(nn.Module):
    def __init__(self, im_size, configs, num_edge_bins=36):
        super(CornerRoomAssociate, self).__init__()
        drn_22 = drn.drn_d_22(pretrained=False)
        drn_modules = list(drn_22.children())
        self.im_size = im_size
        self.configs = configs

        drn_modules[0][0] = nn.Conv2d(configs.input_channels, 16, kernel_size=(7, 7), padding=(3, 3), bias=False)

        self.drn_encoder = nn.Sequential(*drn_modules)
        self.final_linear = nn.Linear(1000, 1 + num_edge_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        The input should contain: corner channel with location and edge orientations,
        mask channel showing the room mask, another mask channel showing masks of other
        channels. Of course the input image should also be a channel.
        :param x: shape K x 256 x 256, K = 1 + 2 + 3 + 3
        :return:
        """
        x = self.drn_encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.final_linear(x)
        preds = self.sigmoid(logits)

        return logits, preds


class CornerCornerAssociate(nn.Module):
    def __init__(self, im_size, configs):
        super(CornerCornerAssociate, self).__init__()
        drn_22 = drn.drn_d_22(pretrained=False)
        drn_modules = list(drn_22.children())
        self.im_size = im_size
        self.configs = configs

        drn_modules[0][0] = nn.Conv2d(configs.input_channels, 16, kernel_size=(7, 7), padding=(3, 3), bias=False)

        self.drn_encoder = nn.Sequential(*drn_modules)
        self.final_linear = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        The input should contain: two corner channels with location and edge orientations,
        mask channel showing the room mask, another mask channel showing masks of other
        channels. Of course the input image should also be a channel.
        :param x: shape K x 256 x 256, K = 1 + 2 + 3 + 3
        :return:
        """
        x = self.drn_encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.final_linear(x)
        preds = self.sigmoid(logits)

        return logits, preds


if __name__ == '__main__':
    model = CornerRoomAssociate(im_size=256, configs=None)

    import numpy as np

    img = torch.Tensor(np.zeros([4, 6, 256, 256]))
    _, y = model(img)
