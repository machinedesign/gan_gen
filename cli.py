from clize import run
from functools import partial
import time
import math
from skimage.io import imsave
import numpy as np
import os
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils

from machinedesign.viz import grid_of_images_default

from model import Gen
from model import AE
from model import Discr
from model import Clf

from utils import Invert
from utils import Gray
from utils import grid_embedding 


def save_weights(m, folder='out'):
    if isinstance(m, nn.Linear):
        w = m.weight.data
        if w.size(1) == 28*28:
            w = w.view(w.size(0), 1, 28, 28)
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/feat.png'.format(folder), gr)
    elif isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        if w.size(1) == 1:
            w = w.view(w.size(0) * w.size(1), w.size(2), w.size(3))
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/feat_{}.png'.format(folder, w.size(0)), gr)
        elif w.size(1) == 3:
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/feat_{}.png'.format(folder, w.size(0)), gr)


def _load_dataset(dataset_name, split='full'):
    if dataset_name == 'mnist':
        dataset = dset.MNIST(
            root='data', 
            download=False,
            transform=transforms.Compose([
                transforms.Scale(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        return dataset
    elif dataset_name == 'shoes':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/shoes/ut-zap50k-images/Shoes',
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'celeba':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/celeba',
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset

    elif dataset_name == 'fonts':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/fonts/'+split,
            transform=transforms.Compose([
            transforms.ToTensor(),
            Invert(),
            Gray(),
         ]))
        return dataset
    else:
        raise ValueError('Error')


def ae(*, folder='out', dataset='celeba'):
    lr = 1e-4
    batch_size = 64
    train = _load_dataset(dataset, split='train')
    trainl = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    x0, _ = train[0]
    nc = x0.size(0)
    ae = AE(nc=nc, latent_size=100)
    ae = ae.cuda()
    opt = optim.Adam(ae.parameters(), lr=lr, betas=(0.5, 0.999))
    nb_epochs = 100
    avg_loss = 0.
    nb_updates = 0
    t0 = time.time()
    for epoch in range(nb_epochs):
        for X, _ in trainl:
            X = Variable(X).cuda()
            ae.zero_grad()
            Xrec, h = ae(X)
            e1 = ((X - Xrec)**2).mean()
            e2 = -((h  - 0.5) ** 2).sum(1).mean()
            e3 = torch.abs(h.mean(0) - 0.5).sum()
            loss = e1 + 0.001*(e2 + e3)
            loss.backward()
            opt.step()
            avg_loss = avg_loss * 0.9 + loss.data[0] * 0.1
            if nb_updates % 100 == 0:
                dt = time.time() - t0
                t0 = time.time()
                print('Epoch {:03d}/{:03d}, Avg loss : {:.6f}, dt : {:.6f}(s)'.format(epoch + 1, nb_epochs, avg_loss, dt))
                im = Xrec.data.cpu().numpy()
                im = grid_of_images_default(im)
                imsave('{}/ae.png'.format(folder), im)
                print(e1.data[0], e2.data[0], e3.data[0])

            if nb_updates % 1000 == 0:
                print(h.data)
            nb_updates += 1
        torch.save(ae, '{}/ae.th'.format(folder))


def clf(*, folder='out', dataset='celeba'):
    lr = 1e-4
    batch_size = 64
    train = _load_dataset(dataset, split='train')
    trainl = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    valid = _load_dataset(dataset_name, split='valid')
    validl = torch.utils.data.DataLoader(
        valid, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=1
    )

    x0, _ = train[0]
    nc = x0.size(0)
    no = 26
    discr = Clf(nc=nc, no=no)
    discr = discr.cuda()
    opt = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))
    nb_epochs = 40
    avg_acc = 0.
    crit = nn.CrossEntropyLoss()
    max_valid_acc = 0
    for epoch in range(nb_epochs):
        for X, y in trainl:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            discr.zero_grad()
            ypred, h = discr(X)
            
            e1 = crit(ypred, y)
            e2 = -((h  - 0.5) ** 2).sum(1).mean()
            e3 = torch.abs(h.mean(1) - 0.5).sum()
            loss = e1 + 0.05*(e2 + e3)
            _, m = ypred.max(1)
            acc = (m == y).float().mean().cpu().data[0]
            avg_acc = avg_acc * 0.9 + acc * 0.1
            loss.backward()
            opt.step()
        accs = []
        for X, y in validl:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            ypred, _ = discr(X)
            _, m = ypred.max(1)
            accs.extend((m==y).float().data.cpu().numpy())
        valid_acc = np.mean(accs)
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save(discr, '{}/clf.th'.format(folder))
        print(h.data[0])
        print('Epoch {:03d}/{:03d}, Avg acc train : {:.3f}, Acc valid : {:.3f}'.format(epoch + 1, nb_epochs, avg_acc, valid_acc))


def train(*, folder='out', dataset='celeba'):
    lr = 0.0002
    nz = 0
    batch_size = 64
    nb_epochs = 300
    wasserstein = True
    dataset = _load_dataset(dataset, split='train')
    x0, _ = dataset[0]
    nc = x0.size(0)
    w = x0.size(1)
    h = x0.size(2)
    _save_weights = partial(save_weights, folder=folder)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )

    encoder = torch.load('{}/ae.th'.format(folder))
    cond = encoder.post_latent[0].weight.size(1)

    act = 'sigmoid' if nc==1 else 'tanh'
    gen = Gen(nz=nz + cond, nc=nc, act=act)
    discr = Discr(nc=nc, act='' if wasserstein else 'sigmoid')

    gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    discr_opt = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))

    input = torch.FloatTensor(batch_size, nc, w, h)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(batch_size)

    if wasserstein:
        real_label = 1
        fake_label = -1
        criterion = lambda output, label:(output*label).mean()
    else:
        real_label = 1
        fake_label = 0
        criterion = nn.BCELoss()

    encoder = encoder.cuda()

    gen = gen.cuda()
    discr =  discr.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)

    for epoch in range(nb_epochs):
        for i, (X, _), in enumerate(dataloader):
            # clamp parameters to a cube
            if wasserstein:
                for p in discr.parameters():
                    p.data.clamp_(-0.01, 0.01)
            # update discriminator
            discr.zero_grad()
            batch_size = X.size(0)
            X = X.cuda()
            input.resize_as_(X).copy_(X)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)
            _, h = encoder(inputv)
            h = h.view(h.size(0), h.size(1), 1, 1)
            h = h.repeat(1, 1, inputv.size(2), inputv.size(3))
            h = (h > 0.5).float()
            print(h)
            #inputv_and_cond = torch.cat((inputv, h), 1)
            inputv_and_cond = inputv
            output = discr(inputv_and_cond)
            errD_real = criterion(output, labelv)
            errD_real.backward()
            D_x = output.data.mean()
            
            if nz > 0:
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise)

            if cond == 0 and nz > 0:
                noise_and_cond = noisev
            elif cond > 0 and nz == 0:
                noise_and_cond = h[:, :, 0:1, 0:1]
            else:
                noise_and_cond = torch.cat((noisev, h[:, :, 0:1, 0:1]), 1)
            fake = gen(noise_and_cond)
            #fake_and_cond = torch.cat((fake, h), 1)
            fake_and_cond = fake

            labelv = Variable(label.fill_(fake_label))
            output = discr(fake_and_cond.detach())
 
            errD_fake = criterion(output, labelv)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            discr_opt.step()
        
            # update generator
            gen.zero_grad()
            fake = gen(noise_and_cond)
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = discr(fake)
            errG = criterion(output, labelv)  + ((fake - inputv)**2).mean()
            errG.backward()
            D_G_z2 = output.data.mean()
            gen_opt.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, nb_epochs, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                x = 0.5 * (X + 1) if act == 'tanh' else X
                f = 0.5 * (fake.data + 1) if act == 'tanh' else fake.data
                vutils.save_image(x, '{}/real_samples.png'.format(folder), normalize=True)
                #fake = gen(fixed_noise)
                vutils.save_image(f, '{}/fake_samples_epoch_{:03d}.png'.format(folder, epoch), normalize=True)
                torch.save(gen, '{}/gen.th'.format(folder))
                torch.save(discr, '{}/discr.th'.format(folder))
                gen.apply(_save_weights)


def extract_codes(*, folder='out', dataset='celeba'):
    batch_size = 64
    exists = set()
    dataset = _load_dataset(dataset, split='full')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    encoder = torch.load('{}/ae.th'.format(folder))
    for X, y in dataloader:
        X = Variable(X).cuda()
        _, h = encoder(X)
        h = (h > 0.5).float()
        h = h.data.cpu().numpy().tolist()
        for hi in h:
            hi = tuple(hi)
            exists.add(hi)
    exists = list(exists)
    exists = np.array(exists)
    print('Size of dataset : {}, Nb of unique codes : {}'.format(len(dataloader), len(exists)))
    np.savez('{}/bin.npz'.format(folder), X=exists)


def gen(*, folder='out', dataset='celeba', way='out_of_distrib'):
    from sklearn.neighbors import BallTree
    from sklearn.manifold import TSNE

    batch_size = 900
    nz = 0
    dataset = _load_dataset(dataset, split='test')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    bin  = np.load('{}/bin.npz'.format(folder))
    bin = bin['X']
    tree = BallTree(bin, metric='hamming')

    exists = set()
    for b in bin:
        b = b.astype('bool')
        b = tuple(b)
        exists.add(b)

    gen = torch.load('{}/gen.th'.format(folder))
    encoder = torch.load('{}/ae.th'.format(folder))
    cond = encoder.post_latent[0].weight.size(1)

    noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).cuda()
    noise = Variable(noise)
    X, y = next(iter(dataloader))
    if way == 'out_of_distrib':
        X = Variable(X).cuda()
        _, h = encoder(X)
        h = (h > 0.5).float()
        sne = TSNE()
        h = h.data.cpu().numpy()
        h2d = sne.fit_transform(h)
        rows = grid_embedding(h2d)
        h = h[rows]
        h = torch.from_numpy(h)
        h = Variable(h).cuda()
    elif way == 'out_of_code':
        h_list = []
        while len(h_list) < batch_size:
            h = (np.random.uniform(size=cond)<=(0.5)).tolist()
            h = tuple(h)
            if h in exists:
                continue
            h_list.append(h)
        h = np.array(h_list).astype('float32')
        sne = TSNE()
        h2d = sne.fit_transform(h)
        rows = grid_embedding(h2d)
        h = h[rows]
        h = torch.from_numpy(h)
        h = Variable(h).cuda()
    else:
        raise ValueError(way)

    if nz == 0:
        noise_and_cond = h.view(h.size(0), h.size(1), 1, 1)
    else:
        noise_and_cond = torch.cat((noise, h.view(h.size(0), h.size(1), 1, 1)), 1)
    fake = gen(noise_and_cond)
    if way == 'out_of_distrib':
        im = fake.data.cpu().numpy()
        sne = TSNE()
        h2d = sne.fit_transform(im.reshape((im.shape[0], -1)))
        rows = grid_embedding(h2d)
        
        im = im[rows]
        im = grid_of_images_default(im, normalize=True)
    
        imsave('{}/check_fake_samples.png'.format(folder), im)

        im = X.data.cpu().numpy()
        im = im[rows]
        im = grid_of_images_default(im, normalize=True)
        imsave('{}/check_true_samples.png'.format(folder), im)

    elif way == 'out_of_code':
        im = fake.data.cpu().numpy()
        sne = TSNE()
        h2d = sne.fit_transform(im.reshape((im.shape[0], -1)))
        rows = grid_embedding(h2d)
        im = im[rows]
        im = grid_of_images_default(im, normalize=True)
        imsave('{}/check_new_samples.png'.format(folder), im)


if __name__ == '__main__':
    run([train, gen, clf, extract_codes, ae])
