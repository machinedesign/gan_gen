from functools import partial
from itertools import chain
import time
import math
import numpy as np
import os
import random
from clize import run

from sklearn.manifold import TSNE

from skimage.io import imsave
from skimage.transform import resize

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torchvision.utils as vutils
from torchvision.models import alexnet

from machinedesign.viz import grid_of_images_default

from model import Gen
from model import AE
from model import Discr
from model import Clf
from model import Pretrained

from utils import Invert
from utils import Gray
from utils import grid_embedding 


def save_weights(m, folder='out', prefix=''):
    if isinstance(m, nn.Linear):
        w = m.weight.data
        if np.sqrt(w.size(1)) == int(w.size(1)):
            s = int(np.sqrst(w.size(1)))
            w = w.view(w.size(0), 1, s, s)
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/{}_feat_{}.png'.format(folder, prefix, w.size(0)), gr)
    elif isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        if w.size(1) == 1:
            w = w.view(w.size(0) * w.size(1), w.size(2), w.size(3))
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/{}_feat_{}.png'.format(folder, prefix, w.size(0)), gr)
        elif w.size(1) == 3:
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/{}_feat_{}.png'.format(folder, prefix, w.size(0)), gr)


def _load_dataset(dataset_name, split='full'):
    if dataset_name == 'mnist':
        dataset = dset.MNIST(
            root='/home/mcherti/work/data/mnist', 
            download=True,
            transform=transforms.Compose([
                transforms.Scale(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        return dataset
    if dataset_name == 'quickdraw':
        X = (np.load('/home/mcherti/work/data/quickdraw/teapot.npy'))
        X = X.reshape((X.shape[0], 28, 28))
        X  = X / 255.
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
        dataset = TensorDataset(X, X)
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
    elif dataset_name == 'birds':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/birds/'+split,
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'sketchy':
        dataset = dset.ImageFolder(root='/home/mcherti/work/data/sketchy/'+split,
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            Gray()
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


def ae(*, folder='out', dataset='celeba', latent_size=100):
    lr = 1e-4
    batch_size = 64
    train = _load_dataset(dataset, split='train')
    _save_weights = partial(save_weights, folder=folder, prefix='ae')
    trainl = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    x0, _ = train[0]
    nc = x0.size(0)
    width = x0.size(2)
    ae = AE(nc=nc, latent_size=latent_size, w=width)
    ae = ae.cuda()
    opt = optim.Adam(ae.parameters(), lr=lr, betas=(0.5, 0.999))
    nb_epochs = 200
    avg_loss = 0.
    avg_e1 = 0.
    avg_e2 = 0.
    avg_e3 = 0.
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
            avg_e1 = avg_e1 * 0.9 + e1.data[0] * 0.1
            avg_e2 = avg_e2 * 0.9 + e2.data[0] * 0.1
            avg_e3 = avg_e3 * 0.9 + e3.data[0] * 0.1

            if nb_updates % 100 == 0:
                dt = time.time() - t0
                t0 = time.time()
                print('Epoch {:03d}/{:03d}, Avg loss : {:.6f}, Avg e1 : {:.6f}, Avg e2 : {:.6f}, Avg e3 : {:.6f}, dt : {:.6f}(s)'.format(epoch + 1, nb_epochs, avg_loss, avg_e1, avg_e2, avg_e3, dt))
                im = Xrec.data.cpu().numpy()
                im = grid_of_images_default(im)
                imsave('{}/ae.png'.format(folder), im)
                ae.apply(_save_weights)

            if nb_updates % 1000 == 0:
                print(h.data)
            nb_updates += 1
        torch.save(ae, '{}/ae.th'.format(folder))


def pretrained(*, folder='out', dataset='celeba', latent_size=200):
    lr = 1e-4
    batch_size = 64
    nb_epochs = 200

    train = _load_dataset(dataset, split='train')
    trainl = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    clf = alexnet(pretrained=True)
    clf = clf.cuda()
    fe = Pretrained(clf.features)
    fe = fe.cuda()
    opt = optim.Adam(chain(fe.encode.parameters(), fe.decode.parameters()), lr=lr, betas=(0.5, 0.999))
    t0 = time.time()
    nb_updates = 0
    avg_loss = 0.
    avg_e1 = 0.
    avg_e2 = 0.
    avg_e3 = 0.
    for epoch in range(nb_epochs):
        for X, _ in trainl:
            X = Variable(X).cuda()
            (htrue, hrec), hbin = fe(X)
            fe.zero_grad()
            e1 = ((htrue - hrec)**2).mean()
            e2 = -((hbin  - 0.5) ** 2).sum(1).mean()
            e3 = torch.abs(hbin.mean(0) - 0.5).sum()
            loss = e1 + 0.005*(e2 + e3)
            loss.backward()
            opt.step()

            avg_loss = avg_loss * 0.9 + loss.data[0] * 0.1
            avg_e1 = avg_e1 * 0.9 + e1.data[0] * 0.1
            avg_e2 = avg_e2 * 0.9 + e2.data[0] * 0.1
            avg_e3 = avg_e3 * 0.9 + e3.data[0] * 0.1
            if nb_updates % 100 == 0:
                dt = time.time() - t0
                t0 = time.time()
                print('Epoch {:03d}/{:03d}, Avg loss : {:.6f}, Avg e1 : {:.6f}, Avg e2 : {:.6f}, Avg e3 : {:.6f}, dt : {:.6f}(s)'.format(epoch + 1, nb_epochs, avg_loss, avg_e1, avg_e2, avg_e3, dt))
            if nb_updates % 1000 == 0:
                print(hbin.data)
            nb_updates += 1
        torch.save(fe, '{}/ae.th'.format(folder))


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


def train(*, folder='out', dataset='celeba', resume=False, wasserstein=True):
    lr = 0.0002
    nz = 0
    batch_size = 64
    nb_epochs = 3000
    dataset = _load_dataset(dataset, split='train')
    x0, _ = dataset[0]
    nc = x0.size(0)
    w = x0.size(1)
    h = x0.size(2)
    _save_weights = partial(save_weights, folder=folder, prefix='gan')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )

    encoder = torch.load('{}/ae.th'.format(folder))
    print(encoder)
    if hasattr(encoder, 'latent_size'):
        cond = encoder.latent_size
    elif hasattr(encoder, 'post_latent'):
        cond = encoder.post_latent[0].weight.size(1)
    else:
        raise ValueError('no cond')

    act = 'sigmoid' if nc==1 else 'tanh'
    if resume:
        gen = torch.load('{}/gen.th'.format(folder))
        discr = torch.load('{}/discr.th'.format(folder))
    else:
        gen = Gen(nz=nz + cond, nc=nc, act=act, w=w)
        discr = Discr(nc=nc, act='' if wasserstein else 'sigmoid', w=w)

    gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    discr_opt = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))

    input = torch.FloatTensor(batch_size, nc, w, h)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
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
    noise = noise.cuda()
    avg_rec = 0.
    for epoch in range(nb_epochs):
        for i, (X, _), in enumerate(dataloader):
            if wasserstein:
                # clamp parameters to a cube
                for p in discr.parameters():
                    p.data.clamp_(-0.01, 0.01)
            
            # Update discriminator
            discr.zero_grad()
            batch_size = X.size(0)
            X = X.cuda()
            input.resize_as_(X).copy_(X)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)
            _, h = encoder(inputv)
            h = (h > 0.5).float()
            h = h.view(h.size(0), h.size(1), 1, 1)
            h = h.repeat(1, 1, inputv.size(2), inputv.size(3))
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
            fake_and_cond = fake

            labelv = Variable(label.fill_(fake_label))
            output = discr(fake_and_cond.detach())
 
            errD_fake = criterion(output, labelv)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            discr_opt.step()
            
            # Update generator
            gen.zero_grad()
            fake = gen(noise_and_cond)
            labelv = Variable(label.fill_(real_label))
            output = discr(fake)
            rec = ((fake - inputv)**2).mean()
            avg_rec = avg_rec * 0.99 + rec.data[0] + 0.01
            errG = criterion(output, labelv)  + rec 
            errG.backward()
            D_G_z2 = output.data.mean()
            gen_opt.step()
            print('{}/{} Rec:{:.6f}'.format(epoch, nb_epochs, rec.data[0]))

            if i % 100 == 0:
                x = 0.5 * (X + 1) if act == 'tanh' else X
                f = 0.5 * (fake.data + 1) if act == 'tanh' else fake.data
                vutils.save_image(x, '{}/real_samples.png'.format(folder), normalize=True)
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
    nb = 0
    for X, y in dataloader:
        X = Variable(X).cuda()
        _, h = encoder(X)
        h = (h > 0.5).float()
        h = h.data.cpu().numpy().tolist()
        for hi in h:
            hi = tuple(hi)
            exists.add(hi)
        nb += len(X)
    exists = list(exists)
    exists = np.array(exists)
    print('Size of dataset : {}, Nb of unique codes : {}'.format(nb, len(exists)))
    np.savez('{}/bin.npz'.format(folder), X=exists)


def gen(*, folder='out', dataset='celeba', way='out_of_distrib'):
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
    exists = set()
    for b in bin:
        b = b.astype('bool')
        b = tuple(b)
        exists.add(b)
    gen = torch.load('{}/gen.th'.format(folder))
    encoder = torch.load('{}/ae.th'.format(folder))

    if hasattr(encoder, 'latent_size'):
        cond = encoder.latent_size
    elif hasattr(encoder, 'post_latent'):
        cond = encoder.post_latent[0].weight.size(1)
    else:
        raise ValueError('no cond')

    noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).cuda()
    noise = Variable(noise)
    X, y = next(iter(dataloader))
    
    # out_of_distrib
    X = Variable(X).cuda()
    _, h = encoder(X)
    h = (h > 0.5).float()
    if nz == 0:
        noise_and_cond = h.view(h.size(0), h.size(1), 1, 1)
    else:
        noise_and_cond = torch.cat((noise, h.view(h.size(0), h.size(1), 1, 1)), 1)
    fake = gen(noise_and_cond)
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
    
    # out_of_code
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
    if nz == 0:
        noise_and_cond = h.view(h.size(0), h.size(1), 1, 1)
    else:
        noise_and_cond = torch.cat((noise, h.view(h.size(0), h.size(1), 1, 1)), 1)
    fake = gen(noise_and_cond)
    im = fake.data.cpu().numpy()
    sne = TSNE()
    h2d = sne.fit_transform(im.reshape((im.shape[0], -1)))
    rows = grid_embedding(h2d)
    im = im[rows]
    im = grid_of_images_default(im, normalize=True)
    imsave('{}/check_new_samples.png'.format(folder), im)

if __name__ == '__main__':
    run([train, gen, clf, extract_codes, ae, pretrained])
