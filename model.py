import torch.nn as nn
from torch.nn.init import xavier_uniform

class Gen(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, act='tanh'):
        super().__init__()
        self.act = act
        self.main = nn.Sequential(
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
        )
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        if self.act == 'tanh':
            out = nn.Tanh()(out)
        elif self.act == 'sigmoid':
            out = nn.Sigmoid()(out)
        return out


class Discr(nn.Module):

    def __init__(self, nc=1, ndf=64, act='sigmoid', no=1):
        super().__init__()
        self.act = act
        self.no = no
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
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
            
            nn.Conv2d(ndf * 8, no, 4, 1, 0, bias=False),
        )
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        if self.act == 'tanh':
            out = nn.Tanh()(out)
        elif self.act == 'sigmoid':
            out = nn.Sigmoid()(out)
        return out.view(-1, self.no)

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

    def __init__(self, nc=1, ndf=64, act='sigmoid', latent_size=256):
        super().__init__()
        self.act = act
        self.ndf = ndf
        self.encode = nn.Sequential(
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
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ndf,      nc, 4, 2, 1, bias=False),
        )
        self.latent = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, latent_size),
            nn.Sigmoid(),
        )
        self.post_latent = nn.Sequential(
            nn.Linear(latent_size, ndf * 8 * 4 * 4)
        )
        self.apply(weights_init)

    def forward(self, input):
        x = self.encode(input)
        x = x.view(x.size(0), self.ndf * 8 * 4 * 4)
        h = self.latent(x)
        x = self.post_latent(h)
        x = x.view(x.size(0), self.ndf * 8, 4, 4)
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

