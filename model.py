import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

class Gen(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, act='tanh', w=64):
        super().__init__()
        self.act = act
        nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ngf * 2**(nb_blocks + 1)
        layers = [
            nn.ConvTranspose2d(nz, nf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf // 2),
                nn.ReLU(True),
            ]) 
            nf = nf // 2
        layers.append(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False)
        )
        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        if self.act == 'tanh':
            out = nn.Tanh()(out)
        elif self.act == 'sigmoid':
            out = nn.Sigmoid()(out)
        return out


class Discr(nn.Module):

    def __init__(self, nc=1, ndf=64, act='sigmoid', no=1, w=64):
        super().__init__()
        self.act = act
        self.no = no

        nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ndf 
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            nf = nf * 2
        layers.append(
            nn.Conv2d(nf, no, 4, 1, 0, bias=False)
        )
        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        if self.act == 'tanh':
            out = nn.Tanh()(out)
        elif self.act == 'sigmoid':
            out = nn.Sigmoid()(out)
        return out.view(-1, self.no)


class Pretrained(nn.Module):

    def __init__(self, features, h_size=256, latent_size=256):
        super().__init__()
        self.latent_size = latent_size
        self.features = features
        self.encode = nn.Sequential(
            nn.Linear(h_size, latent_size),
            nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.Linear(latent_size, h_size),
            nn.ReLU(True)
        )
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
        self.clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()

    def forward(self, input):
        input = norm(input, self.clf_mean, self.clf_std)
        x = input
        h = self.features(x)
        h = h.detach()
        h = h.view(h.size(0), -1)
        htrue = h
        h = self.encode(h)
        hbin = h
        hrec = self.decode(h)
        return (htrue, hrec), hbin


def norm(x, m, s):
    x = (x + 1) / 2.
    x = x - m.repeat(x.size(0), 1, x.size(2), x.size(3))
    x = x / s.repeat(x.size(0), 1, x.size(2), x.size(3))
    return x



class Clf(nn.Module):

    def __init__(self, nc=1, ndf=64, act='sigmoid', no=1):
        super().__init__()
        self.act = act
        self.no = no
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, 100),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(100, no)
        )
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, self.ndf * 8 * 4 * 4)
        h = self.fc(out)
        out = self.out(h)
        return out, h


class AE(nn.Module):

    def __init__(self, nc=1, ndf=64, act='sigmoid', latent_size=256, w=64):
        super().__init__()
        self.latent_size = latent_size
        self.act = act
        self.ndf = ndf

        nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ndf 
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            nf = nf * 2
        
        self.encode = nn.Sequential(*layers)

        wl = w // 2**(nb_blocks+1)
        self.latent = nn.Sequential(
            nn.Linear(nf * wl * wl, latent_size),
            nn.Sigmoid(),
        )
        self.post_latent = nn.Sequential(
            nn.Linear(latent_size, nf * wl * wl)
        )
        layers = []
        for _ in range(nb_blocks):
            layers.extend([
                nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf // 2),
                nn.ReLU(True),
            ])
            nf = nf // 2
        layers.append(
            nn.ConvTranspose2d(nf,  nc, 4, 2, 1, bias=False)
        )
        self.decode = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        x = self.encode(input)
        pre_latent_size = x.size()
        x = x.view(x.size(0), -1)
        h = self.latent(x)
        x = self.post_latent(h)
        x = x.view(pre_latent_size)
        xrec = self.decode(x)
        return xrec, h


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == 'Linear':
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)

