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


class SparseGen(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(     nz, ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        ) 
        self.layer1_out  = nn.Sequential(    
            nn.ConvTranspose2d(    ngf*8,      nc, 16, 16, 0, bias=False),
            nn.Tanh()
        )
        self.layer1_out.name = 'layer1'
        self.layer2_out  = nn.Sequential(    
            nn.ConvTranspose2d(    ngf*4,      nc, 8, 8, 0, bias=False),
            nn.Tanh()
        )
        self.layer2_out.name = 'layer2'
        self.layer3_out  = nn.Sequential(    
            nn.ConvTranspose2d(    ngf*2,      nc, 4, 4, 0, bias=False),
            nn.Tanh()
        )
        self.layer3_out.name = 'layer3'
        self.apply(weights_init)

    def forward(self, input):
        x1 = self.layer1(input)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        #x1t = spatial_sparsity(x1)
        #x2t = spatial_sparsity(x2)
        #x3t = spatial_sparsity(x3)
        x1t = x1
        x2t = x2
        x3t = x3
        o1 = self.layer1_out(x1t)
        o2 = self.layer2_out(x2t)
        o3 = self.layer3_out(x3t)
        o = 1/3.*(o1+o2+o3)
        return o

def spatial_sparsity(x):
    xf = x.view(x.size(0), x.size(1), -1)
    m, _ = xf.max(2)
    m = m.repeat(1, 1, xf.size(2))
    xf = xf * (xf==m).float()
    xf = xf.view(x.size())
    return xf


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
            nn.Linear(ndf * 8 * 4 * 4, 20),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(20, no)
        )
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, self.ndf * 8 * 4 * 4)
        h = self.fc(out)
        out = self.out(h)
        return out, h


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

