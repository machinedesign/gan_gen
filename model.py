import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

from light_cnn import LightCNN_29Layers_v2

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


class PPGen(nn.Module):
    def __init__(self, nz=4096, act='tanh'):
        super().__init__()
        self.act = act
        self.fc = nn.Sequential(
            nn.Linear(nz,  4096), #defc7
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.3),
            nn.Linear(4096, 4096), #defc6
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.3),
            nn.Linear(4096, 4096), #defc5
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.3),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), #deconv5
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), #conv5_1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.3),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), #decon4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False), #conv4_1
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), #deconv3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), #conv3_1
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
         
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), #deconv2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), #deconv1
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.3),
            
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False), #deconv0
        )
        self.apply(weights_init)

    def forward(self, input):
        x = input.view(input.size(0), input.size(1))
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.conv(x)
        if self.act == 'tanh':
            x = nn.Tanh()(x)
        elif self.act == 'sigmoid':
            x = nn.Sigmoid()(x)
        return x


class PPDiscr(nn.Module):

    def __init__(self, act='sigmoid'):
        super().__init__()
        self.act = act
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )
        self.apply(weights_init)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        if self.act == 'tanh':
            x = nn.Tanh()(x)
        elif self.act == 'sigmoid':
            x = nn.Sigmoid()(x)
        return x[:, 0]


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
        if input.size(1) == 3:
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


class Resize(nn.Module):

    def __init__(self, size=2, nc=3, w=256):
        super().__init__()
        self.pool = nn.AvgPool2d(size)
        self.latent_size = ((w * w) // (size*size)) * nc
        print(self.latent_size)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return None, x


class PretrainedFrozen(nn.Module):

    def __init__(self, features, classifier, h_size=4096):
        super().__init__()
        self.features = features
        self.classifier = classifier
        self.latent_size = h_size
        self.h_size = h_size
        self.encode = nn.Sequential()
        self.decode = nn.Sequential()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
        self.clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()

    def forward(self, input):
        input = norm(input, self.clf_mean, self.clf_std)
        x = input
        x = x[:, :, 16:-16, 16:-16]
        h = self.features(x)
        h = h.detach()
        h = h.view(h.size(0), -1)
        h = self.classifier[0](h)
        h = self.classifier[1](h)
        h = self.classifier[2](h)
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
            nn.Linear(ndf * 8 * 4 * 4, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(512, no)
        )
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, self.ndf * 8 * 4 * 4)
        h = self.fc(out)
        out = self.out(h)
        return out, h


class AE(nn.Module):

    def __init__(self, nc=1, ndf=64, act='sigmoid', latent_size=256, w=64, round=False):
        super().__init__()
        self.latent_size = latent_size
        self.act = act
        self.ndf = ndf
        self.round = round

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
        if self.round:
            h = h.round()
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


class FaceDescriptor:

    def __init__(self, path="LightCNN_29Layers_V2_checkpoint.pth.tar", device="cpu"):
        model = LightCNN_29Layers_v2(num_classes=80013)
        checkpoint = torch.load(path, map_location="cpu")
        ck = checkpoint['state_dict']
        ck_ = {}
        for k, v in ck.items():
            ck_[k.replace("module.", "")] = v
        model.load_state_dict(ck_)
        model.to(device)
        self.net = model
        self.latent_size = 128

    def forward(self, x):
        x = (x + 1) / 2.
        x = x.mean(dim=1, keepdim=True) #grayscale
        x = nn.AdaptiveAvgPool2d((128, 128))(x) #resize to 128x128
        return self.net(x)
