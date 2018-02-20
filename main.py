'''Train a VAE in CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image


import os
import argparse
import numpy as np

from models import *
from utils import *


# arguments
parser = argparse.ArgumentParser(description='VAE CIFAR-10 Example')

parser.add_argument('--batch-size', type=int, default=4*256, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--dataset', type=str, default='CIFAR-10',
                    help='select between CIFAR-10 and MNIST')

args = parser.parse_args()

# seed
torch.manual_seed(args.seed)

# use cuda
use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == 'CIFAR-10':

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_channels = 3
    woh = 32

else:
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True,
        transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    num_channels = 1
    woh = 28


# model
model = basic_model()#fc_model(num_channels, woh)
if use_cuda:
    model.cuda()
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())) # replicates the model in all the GPUs. The batch size should be a multiple of the number of GPUs
    #cudnn.benchmark = True # interesting if the size of the inputs is constant. cudnn will use the best altgorithm for the size of the input encounter. If a new size is encounter this optimation takes again place, leading to a worse runtime
    

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    L1Loss = nn.L1Loss()
    l1loss = L1Loss(recon_x, x)
    #BCE = F.binary_cross_entropy(recon_x.view(-1,num_flat_features(x)), x.view(-1, num_flat_features(x))) # The binary cross-entropy only has sense in a binary image

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= args.batch_size * (num_channels*woh*woh)

    #print('BCE:', BCE, ' KLD:', KLD)
    
    return l1loss + KLD


# train function
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if use_cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# test function
def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, num_channels, woh, woh)[:n]])
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

# main loop
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    #sample = Variable(torch.randn(64, 200))
    #if use_cuda:
    #    sample = sample.cuda()
    #sample = model.decode(sample).cpu() # this line rises an error if the model is parallelized
    #save_image(sample.data.view(64, num_channels, woh, woh),
    #           'results/sample_' + str(epoch) + '.png')
