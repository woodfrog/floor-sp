import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class ResidualModule(nn.Module):
    def __init__(self, modeltype, indim, hiddim, outdim, nlayers, nres, ifgate=False, nonlinear='elu'):
        super(ResidualModule, self).__init__()
        if ifgate:
            print('Using gated version.')
        if modeltype == 'encoder':
            self.model = self.encoder(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear)
        elif modeltype == 'decoder':
            self.model = self.decoder(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear)
        elif modeltype == 'plain':
            self.model = self.plain(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear)
        else:
            raise ('Uknown model type.')

    def encoder(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear):
        layers = []
        layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim))

        for i in range(0, nres):
            for j in range(0, nlayers):
                layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim))
            layers.append(ResidualBlock('down', nonlinear, ifgate, hiddim, hiddim))

        layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim))

        return nn.Sequential(*layers)

    def decoder(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear):
        layers = []
        layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim))

        for i in range(0, nres):
            for j in range(0, nlayers):
                layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim))
            layers.append(ResidualBlock('up', nonlinear, ifgate, hiddim, hiddim))

        layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim))

        return nn.Sequential(*layers)

    def plain(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear):
        layers = []
        layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim))

        for i in range(0, nres):
            for j in range(0, nlayers):
                layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim))

        layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, resample, nonlinear, ifgate, indim, outdim):
        super(ResidualBlock, self).__init__()

        self.ifgate = ifgate
        self.indim = indim
        self.outdim = outdim
        self.resample = resample

        if resample == 'down':
            convtype = 'sconv_d'
        elif resample == 'up':
            convtype = 'upconv'
        elif resample == None:
            convtype = 'sconv'

        self.shortflag = False
        if not (indim == outdim and resample == None):
            self.shortcut = self.conv(convtype, indim, outdim)
            self.shortflag = True

        if ifgate:
            self.conv1 = nn.Conv2d(indim, outdim, 3, 1, 1)
            self.conv2 = nn.Conv2d(indim, outdim, 3, 1, 1)
            self.c = nn.Sigmoid()
            self.g = nn.Tanh()
            self.conv3 = self.conv(convtype, outdim, outdim)
            self.act = self.nonlinear(nonlinear)
        else:
            self.resblock = nn.Sequential(
                self.conv('sconv', indim, outdim),
                nn.BatchNorm2d(outdim),
                self.nonlinear(nonlinear),
                self.conv(convtype, outdim, outdim),
                nn.BatchNorm2d(outdim),
                self.nonlinear(nonlinear)
            )

    def conv(self, name, indim, outdim):
        if name == 'sconv_d':
                return nn.Conv2d(indim, outdim, 4, 2, 1)
        elif name == 'sconv':
                return nn.Conv2d(indim, outdim, 3, 1, 1)
        elif name == 'upconv':
                return nn.ConvTranspose2d(indim, outdim, 4, 2, 1)
        else:
            raise ("Unknown convolution type")

    def nonlinear(self, name):
        if name == 'elu':
            return nn.ELU(1, True)
        elif name == 'relu':
            return nn.ReLU(True)

    def forward(self, x):
        if self.ifgate:
            conv1 = self.conv1(x)
            conv2 = self.conv2(x)
            c = self.c(conv1)
            g = self.g(conv2)
            gated = c * g
            conv3 = self.conv3(gated)
            res = self.act(conv3)
            if not (self.indim == self.outdim and self.resample == None):
                out = self.shortcut(x) + res
            else:
                out = x + res
        else:
            if self.shortflag:
                out = self.shortcut(x) + self.resblock(x)
            else:
                out = x + self.resblock(x)

        return out
