import _init_paths
import torch
import torch.nn as nn

from models import drn
from models.residual import ResidualModule

class CornerNet(nn.Module):
    def __init__(self, num_input_channel, base_pretrained, corner_bin_size, im_size, configs):
        super(CornerNet, self).__init__()
        drn_22 = drn.drn_d_22(pretrained=base_pretrained)
        base_modules = list(drn_22.children())

        # Sometimes we need to add extra input channels (more than 3 channels for image)
        base_modules[0][0] = nn.Conv2d(num_input_channel, 16, kernel_size=(7, 7), padding=(3, 3),
                                       bias=False)
        drn_encoder = nn.Sequential(*base_modules[:-2])

        self.encoder = drn_encoder
        # todo: drn encoder output size 32, need to do up-sampling for 3 times using transposed conv, might be more flexible here?
        self.encoder_out_dim = 512

        self.out_channels = corner_bin_size + 1
        self.decoder = ResidualModule(modeltype='decoder', indim=self.encoder_out_dim,
                                      hiddim=self.encoder_out_dim // 2, outdim=self.encoder_out_dim // 4, nlayers=1,
                                      nres=3, ifgate=False, nonlinear='relu')
        self.final_layer = nn.Conv2d(self.encoder_out_dim // 4, self.out_channels, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.im_size = im_size

    def forward(self, img):
        encoding = self.encoder(img)
        out = self.decoder(encoding)
        corner_pred_logits = self.final_layer(out)
        corner_pred = self.sigmoid(corner_pred_logits)
        return corner_pred_logits, corner_pred


class CornerEdgeNet(nn.Module):
    """
        A  two-head network which predicts both corner and edge maps with high-level feature sharing
    """
    def __init__(self, num_input_channel, base_pretrained, bin_size, im_size, configs):
        super(CornerEdgeNet, self).__init__()
        drn_22 = drn.drn_d_22(pretrained=base_pretrained)
        base_modules = list(drn_22.children())

        # Sometimes we need to add extra input channels (more than 3 channels for image)
        base_modules[0][0] = nn.Conv2d(num_input_channel, 16, kernel_size=(7, 7), padding=(3, 3),
                                       bias=False)
        drn_encoder = nn.Sequential(*base_modules[:-2])

        self.encoder = drn_encoder
        # todo: drn encoder output size 32, need to do up-sampling for 3 times using transposed conv, might be more flexible here?
        self.encoder_out_dim = 512

        self.out_channels = bin_size + 1
        self.decoder_base = ResidualModule(modeltype='decoder', indim=self.encoder_out_dim,
                                      hiddim=self.encoder_out_dim // 2, outdim=self.encoder_out_dim // 4, nlayers=1,
                                      nres=2, ifgate=False, nonlinear='relu')
        self.decode_edge = ResidualModule(modeltype='decoder', indim=self.encoder_out_dim // 4,
                                      hiddim=self.encoder_out_dim // 8, outdim=self.encoder_out_dim // 16, nlayers=1,
                                      nres=1, ifgate=False, nonlinear='relu')
        self.decode_corner = ResidualModule(modeltype='decoder', indim=self.encoder_out_dim // 4,
                                          hiddim=self.encoder_out_dim // 8, outdim=self.encoder_out_dim // 16, nlayers=1,
                                          nres=1, ifgate=False, nonlinear='relu')

        self.final_layer_edge = nn.Conv2d(self.encoder_out_dim // 16, self.out_channels, kernel_size=(1, 1))
        self.final_layer_coorner = nn.Conv2d(self.encoder_out_dim // 16, self.out_channels, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.im_size = im_size

    def forward(self, img):
        encoding = self.encoder(img)
        shared_feature = self.decoder_base(encoding)
        feature_edge = self.decode_edge(shared_feature)
        feature_corner = self.decode_corner(shared_feature)
        edge_pred_logits = self.final_layer_edge(feature_edge)
        edge_pred = self.sigmoid(edge_pred_logits)
        corner_pred_logits = self.final_layer_coorner(feature_corner)
        corner_pred = self.sigmoid(corner_pred_logits)
        return corner_pred_logits, edge_pred_logits, edge_pred, corner_pred


if __name__ == '__main__':
    model = CornerNet(corner_bin_size=20)

    import numpy as np

    img = torch.Tensor(np.zeros([1, 3, 256, 256]))
    x = model(img)
