import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from functools import partial
from itertools import chain
import time
import numpy as np
import os
from clize import run

from sklearn.manifold import TSNE
from sklearn.neighbors import BallTree

from skimage.io import imsave

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision.models import alexnet
from utils import grid_of_images_default

from model import Gen
from model import AE
from model import Discr
from model import Clf
from model import Pretrained
from model import PretrainedFrozen
from model import PPGen
from model import PPDiscr
from model import norm
from model import Resize
from model import FaceDescriptor

from utils import grid_embedding 

from data import load_dataset


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


def face_descriptor(*, folder='out'):
    face_desc = FaceDescriptor()
    torch.save(face_desc, os.path.join(folder, 'ae.th'))


def ae(*, folder='out', dataset='celeba', latent_size=100, round=False, device="cuda"):
    lr = 1e-4
    batch_size = 64
    train = load_dataset(dataset)
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
    ae = AE(nc=nc, latent_size=latent_size, w=width, round=round)
    ae = ae.to(device)
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
            X = Variable(X).to(device)
            ae.zero_grad()
            Xrec, h = ae(X)
            e1 = ((X - Xrec)**2).mean()
            e2 = -((h  - 0.5) ** 2).sum(1).mean()
            e3 = torch.abs(h.mean(0) - 0.5).sum()
            loss = e1
            loss.backward()
            opt.step()
            avg_loss = avg_loss * 0.9 + loss.item() * 0.1
            avg_e1 = avg_e1 * 0.9 + e1.item() * 0.1
            avg_e2 = avg_e2 * 0.9 + e2.item() * 0.1
            avg_e3 = avg_e3 * 0.9 + e3.item() * 0.1
            if nb_updates % 100 == 0:
                dt = time.time() - t0
                t0 = time.time()
                print('Epoch {:03d}/{:03d}, Avg loss : {:.6f}, Avg e1 : {:.6f}, Avg e2 : {:.6f}, Avg e3 : {:.6f}, dt : {:.6f}(s)'.format(epoch + 1, nb_epochs, avg_loss, avg_e1, avg_e2, avg_e3, dt))
                im = Xrec.data.cpu().numpy()
                im = grid_of_images_default(im, normalize=True)
                imsave('{}/ae.png'.format(folder), im)
                ae.apply(_save_weights)

            if nb_updates % 1000 == 0:
                print(h.data)
            nb_updates += 1
        torch.save(ae, '{}/ae.th'.format(folder))


def pretrained_frozen(*, folder='out', h_size=4096, device="cuda"):
    clf = alexnet(pretrained=True)
    clf = clf.to(device)
    clf = PretrainedFrozen(features=clf.features, classifier=clf.classifier, h_size=h_size)
    torch.save(clf, os.path.join(folder, 'ae.th'))

def resize(*, folder='out', size=2, nc=3, w=256):
    ae = Resize(size, nc=nc, w=w)
    torch.save(ae, os.path.join(folder, 'ae.th'))
    

def pretrained(*, folder='out', dataset='celeba', latent_size=200, constraint='binary', h_size=256, classifier='alexnet', device="cuda"):
    lr = 1e-4
    batch_size = 64
    nb_epochs = 200

    train = load_dataset(dataset, split='train')
    trainl = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    if classifier == 'alexnet':
        clf = alexnet(pretrained=True)
        clf = clf.to(device)
    else:
        clf = torch.load(classifier)
        clf = clf.to(device)
    features = clf.features if hasattr(clf, 'features') else clf.main
    fe = Pretrained(features, latent_size=latent_size, h_size=h_size)
    fe = fe.to(device)
    opt = optim.Adam(chain(fe.encode.parameters(), fe.decode.parameters()), lr=lr, betas=(0.5, 0.999))
    t0 = time.time()
    nb_updates = 0
    avg_loss = 0.
    avg_e1 = 0.
    avg_e2 = 0.
    avg_e3 = 0.
    for epoch in range(nb_epochs):
        for X, _ in trainl:
            X = Variable(X).to(device)
            (htrue, hrec), hbin = fe(X)
            fe.zero_grad()
            if constraint == 'binary':
                e1 = ((htrue - hrec)**2).mean()
                e2 = -((hbin  - 0.5) ** 2).sum(1).mean()
                e3 = torch.abs(hbin.mean(0) - 0.5).sum()
                loss = e1 + 0.01 * (e2 + e3)
            elif constraint == 'binary_sparse':
                e1 = ((htrue - hrec)**2).mean()
                e2 = -((hbin  - 0.5) ** 2).sum(1).mean()
                e3 = torch.abs(hbin.mean(0)).sum()
                loss = e1 + 0.001 * (e2 + e3)
            elif constraint == '':
                e1 = ((htrue - hrec)**2).mean()
                e2 = e1
                e3 = e1
                loss = e1
            loss.backward()
            opt.step()
            avg_loss = avg_loss * 0.9 + loss.item() * 0.1
            avg_e1 = avg_e1 * 0.9 + e1.item() * 0.1
            avg_e2 = avg_e2 * 0.9 + e2.item() * 0.1
            avg_e3 = avg_e3 * 0.9 + e3.item() * 0.1
            if nb_updates % 100 == 0:
                dt = time.time() - t0
                t0 = time.time()
                print('Epoch {:03d}/{:03d}, Avg loss : {:.6f}, Avg e1 : {:.6f}, Avg e2 : {:.6f}, Avg e3 : {:.6f}, dt : {:.6f}(s)'.format(epoch + 1, nb_epochs, avg_loss, avg_e1, avg_e2, avg_e3, dt))
            if nb_updates % 1000 == 0:
                print(hbin.data)
            nb_updates += 1
        torch.save(fe, '{}/ae.th'.format(folder))


def clf(*, folder='out', dataset='celeba', no=26, device="cuda"):
    lr = 1e-4
    batch_size = 64
    train = load_dataset(dataset, split='train')
    trainl = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    valid = load_dataset(dataset, split='valid')
    validl = torch.utils.data.DataLoader(
        valid, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=1
    )

    x0, _ = train[0]
    nc = x0.size(0)
    discr = Clf(nc=nc, no=no)
    discr = discr.to(device)
    opt = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))
    nb_epochs = 40
    avg_acc = 0.
    crit = nn.CrossEntropyLoss()
    max_valid_acc = 0
    for epoch in range(nb_epochs):
        for X, y in trainl:
            X = Variable(X).to(device)
            y = Variable(y).to(device)
            discr.zero_grad()
            ypred, h = discr(X)
            e1 = crit(ypred, y)
            loss = e1
            _, m = ypred.max(1)
            acc = (m == y).float().mean().cpu().item()
            avg_acc = avg_acc * 0.9 + acc * 0.1
            loss.backward()
            opt.step()
        accs = []
        for X, y in validl:
            X = Variable(X).to(device)
            y = Variable(y).to(device)
            ypred, _ = discr(X)
            _, m = ypred.max(1)
            accs.extend((m==y).float().data.cpu().numpy())
        valid_acc = np.mean(accs)
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save(discr, '{}/clf.th'.format(folder))
        print('Epoch {:03d}/{:03d}, Avg acc train : {:.3f}, Acc valid : {:.3f}'.format(epoch + 1, nb_epochs, avg_acc, valid_acc))


def train(*, folder='out', dataset='celeba', resume=False, wasserstein=True, binarize=False,
          batch_size=64, nz=0, model=None, device="cuda"):
    lr = 0.0002
    nb_epochs = 3000
    dataset = load_dataset(dataset, split='train')
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
        if model is None:
            gen = Gen(nz=nz + cond, nc=nc, act=act, w=w)
            discr = Discr(nc=nc, act='' if wasserstein else 'sigmoid', w=w)
        elif model == 'ppgn':
            gen = PPGen(nz=nz + cond, act=act)
            discr = PPDiscr(act='' if wasserstein else 'sigmoid')
    if wasserstein:
        gen_opt = optim.RMSprop(gen.parameters(), lr=lr)
        discr_opt = optim.RMSprop(discr.parameters(), lr=lr)
    else:
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

    encoder = encoder.to(device)

    gen = gen.to(device)
    discr =  discr.to(device)
    input, label = input.to(device), label.to(device)
    noise = noise.to(device)

    stats = defaultdict(list)
    nb_updates = 0
    for epoch in range(nb_epochs):
        for i, (X, _), in enumerate(dataloader):
            if wasserstein:
                # clamp parameters to a cube
                for p in discr.parameters():
                    p.data.clamp_(-0.01, 0.01)
            # Update discriminator
            discr.zero_grad()
            batch_size = X.size(0)
            X = X.to(device)
            input.resize_as_(X).copy_(X)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)
            _, h = encoder(inputv)
            h_real = h
            if binarize:
                h = (h > 0.5).float()
            h = h.view(h.size(0), h.size(1), 1, 1)
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
            discr_opt.step()
            # Update generator
            gen.zero_grad()
            fake = gen(noise_and_cond)
            labelv = Variable(label.fill_(real_label))
            output = discr(fake)
            rec = ((fake - inputv)**2).mean()

            _, h_fake = encoder(fake)
            h_rec = ((h_fake - h_real)**2).mean()

            g = criterion(output, labelv)
            errG = 0.001 * g + 3 * rec + 0.001 * h_rec
            errG.backward()

            gen_opt.step()
            print('{}/{} Rec:{:.6f} Hrec : {:.6f} dreal : {:.6f} dfake : {:.6f}'.format(
                epoch, 
                nb_epochs, 
                rec.item(),
                h_rec.item(), 
                D_x, D_G_z1
            ))
            nb_updates += 1
            stats['iter'].append(nb_updates)
            stats['rec'].append(rec.item())
            stats['h_rec'].append(h_rec.item())
            stats['discr_real'].append(errD_real.item())
            stats['discr_fake'].append(errD_fake.item())

            if nb_updates % 100 == 0:
                x = 0.5 * (X + 1) if act == 'tanh' else X
                f = 0.5 * (fake.data + 1) if act == 'tanh' else fake.data
                vutils.save_image(x, '{}/real_samples.png'.format(folder), normalize=True)
                vutils.save_image(f, '{}/fake_samples_epoch_{:03d}.png'.format(folder, epoch), normalize=True)
                torch.save(gen, '{}/gen.th'.format(folder))
                torch.save(discr, '{}/discr.th'.format(folder))
                torch.save(clf, '{}/clf.th'.format(folder))
                gen.apply(_save_weights)
                pd.DataFrame(stats).to_csv('{}/stats.csv'.format(folder))


def extract_codes(*, folder='out', dataset='celeba', device="cuda"):
    batch_size = 64
    exists = set()
    dataset = load_dataset(dataset, split='full')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    encoder = torch.load('{}/ae.th'.format(folder))
    nb = 0
    for X, y in dataloader:
        X = Variable(X).to(device)
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


def cluster(*, folder='out', dataset='celeba', device="cuda"):
    batch_size = 900
    ae = torch.load('{}/ae.th'.format(folder))
    dataset = load_dataset(dataset, split='test')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    X, _ = next(iter(dataloader))
    im = X.numpy()
    X = Variable(X).to(device)
    _, h = ae(X)
    h = (h>0.5).float()
    h = h.data.cpu().numpy()
    sne = TSNE()
    h2d = sne.fit_transform(h)
    rows = grid_embedding(h2d)
    im = im[rows]
    im = grid_of_images_default(im, normalize=True)
    imsave('{}/codes.png'.format(folder), im)
 
def ppgn(*, folder='out', dataset='celeba', nb_examples=1, unit=0, device="cuda"):
    gen = torch.load('{}/gen.th'.format(folder))
    gen.train()
    encoder = torch.load('{}/ae.th'.format(folder))
    clf = alexnet(pretrained=True)
    clf = clf.to(device)
    if hasattr(encoder, 'latent_size'):
        cond = encoder.latent_size
    elif hasattr(encoder, 'post_latent'):
        cond = encoder.post_latent[0].weight.size(1)
    else:
        raise ValueError('no cond')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).to(device)
    std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).to(device)
    
    grads = {}
    def save_grads(g):
        grads['h'] = g
    
    clfImageSize = 224
    def prep(x):
        if x.size(2) != 256:
            x = torch.nn.UpsamplingBilinear2d(scale_factor=256//x.size(2))(x)
        p = 256 - clfImageSize
        if p > 0:
            x = x[:, :, p//2:-p//2, p//2:-p//2]
        return x
    
    h = torch.rand((nb_examples, cond, 1, 1))
    h = h.to(device)

    for i in range(100):
        hv = Variable(h, requires_grad=True)
        hv.register_hook(save_grads)
        x = gen(hv)
        generated = x
        _, hrec = encoder(x)
        hrec = hrec.view(hrec.size(0), hrec.size(1), 1, 1)
        x = prep(x)
        x = norm(x, mean, std)
        y = clf(x)
        loss = y[:, unit].mean()
        loss.backward()
        g = grads['h']
        h += g.data + 0.1 * (hrec.data - h)
        h.clamp_(0, 1)
        #h = (h > 0.5).float()
        print(loss.item())
    
    im = (generated.data + 1) / 2
    im = im.cpu().numpy()
    im = grid_of_images_default(im)
    imsave('{}/ppgn.png'.format(folder), im)

 
def gen(*, folder='out', dataset='celeba', device="cuda"):
    batch_size = 2500
    nz = 0
    dataset = load_dataset(dataset, split='test')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    if not os.path.exists('{}/bin.npz'.format(folder)):
        extract_codes(folder=folder, dataset=dataset)
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

    noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
    noise = Variable(noise)
    X, y = next(iter(dataloader))
    
    # out_of_distrib
    X = Variable(X[0:900]).to(device)
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
    fake_images = im
    im = grid_of_images_default(im, normalize=True)
    imsave('{}/check_fake_samples.png'.format(folder), im)
    im = X.data.cpu().numpy()
    im = im[rows]
    real_images = im
    im = grid_of_images_default(im, normalize=True)
    imsave('{}/check_true_samples.png'.format(folder), im)
    # out_of_code
    tree = BallTree(bin, metric='hamming')
    h_new = []
    h_train = []
    while len(h_new) < batch_size:
        h = (np.random.uniform(size=cond)<=(0.5)).tolist()
        h = tuple(h)
        dist, ind = tree.query([h], k=1)
        dist = dist[0, 0]
        ind = ind[0, 0]
        if dist == 0:
            continue
        h_new.append(h)
        h_train.append(bin[ind])

    h_new = np.array(h_new).astype('float32')
    h_train = np.array(h_train).astype('float32')

    def forward(h):
        h = torch.from_numpy(h)
        h = Variable(h).to(device)
        if nz == 0:
            noise_and_cond = h.view(h.size(0), h.size(1), 1, 1)
        else:
            noise_and_cond = torch.cat((noise, h.view(h.size(0), h.size(1), 1, 1)), 1)
        fake = gen(noise_and_cond)
        im = fake.data.cpu().numpy()

        _, h_fake = encoder(fake)
        return im
    
    im_new = minibatcher(forward, h_new)
    im_train = minibatcher(forward, h_train)

    sne = TSNE()
    h2d = sne.fit_transform(im_new.reshape((im_new.shape[0], -1)))
    rows = grid_embedding(h2d)
    im_new = im_new[rows]
    new_images = im_new
    im_train = im_train[rows]
    np.savez('{}/im_new.npz'.format(folder), X=im_new)
    im_new = grid_of_images_default(im_new, normalize=True)
    imsave('{}/check_new_samples.png'.format(folder), im_new)
    im_train = grid_of_images_default(im_train, normalize=True)
    imsave('{}/check_new_samples_nearest_on_data.png'.format(folder), im_train)

    # TSNE
    # real_images, fake_images, new_images, nearest_train_images
    clf = alexnet(pretrained=True)
    clf = clf.to(device)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).to(device)
    clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).to(device)
    def enc(X):
        if X.shape[1] == 1:
            return X.reshape((X.shape[0], -1))
        else:
            X = torch.from_numpy(X)
            X = Variable(X)
            X = X.to(device)
            X = norm(X, clf_mean, clf_std)
            h = clf.features(X)
            h = h.view(h.size(0), -1)
            h = h.data.cpu().numpy()
            return h
    h_real_images = []
    for i in range(0, real_images.shape[0], 64):
        h_real_images.append(enc(real_images[i:i+64]))
    h_real_images = np.concatenate(h_real_images, axis=0)
    
    h_list  = [
        minibatcher(enc, real_images),
        minibatcher(enc, fake_images),
        minibatcher(enc, new_images)
    ]
    labels = np.array(
        [0] * real_images.shape[0] + 
        [1] * fake_images.shape[0] + 
        [2] * new_images.shape[0]
    )
    H = np.concatenate(h_list, axis=0)
    sne = TSNE(perplexity=50)
    H = sne.fit_transform(H)
    
    fig = plt.figure(figsize=(10, 10))
    colors = [
        'r',
        'g',
        'orange',
    ]
    caption = [
        'real',
        'fake',
        'new'
    ]
    for label in (0, 1, 2):
        g = labels == label
        plt.scatter(H[g, 0], H[g, 1],
                    marker='+', c=colors[label], s=40, alpha=0.7,
                    label=caption[label])
    plt.legend()
    plt.savefig('{}/sne.png'.format(folder))
    plt.close(fig)
   

def minibatcher(f, X, batch_size=64):
    h = []
    for i in range(0, len(X), batch_size):
        h.append(f(X[i:i+batch_size]))
    return np.concatenate(h, axis=0)


if __name__ == '__main__':
    run([train, gen, clf, extract_codes, ae, pretrained, cluster, pretrained_frozen, ppgn, resize, face_descriptor])
