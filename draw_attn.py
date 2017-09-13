#need to give --batch-size
from __future__ import print_function
import argparse
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io
import string
import numpy as np
import torch
import torch.utils.data 

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#args.cuda = args.no_cuda


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)



class draw(nn.Module):
    def __init__(self, input_size, patch_size, A, B, N,  seq_len, batch_size):
        super(draw, self).__init__()
        #self.read()
        #self.encoder_RNN()
        #self.Q() #mu, sigma, includes reparametrization etc.
        #self.sample()
        #self.decoder_RNN()
        #self.write()

        self.input_size = input_size 
        self.patch_size = patch-size 
        self.A = A
        self.B = B
        self.N = N
        self.seq_len = seq_len
        self.dec_hidden_size = patch_size
        self.enc_hidden_size = patch_size
        self.hidden_size = patch_size 
        self.z_size = int(self.hidden_size/2)
        self.batch_size = batch_size

        #writer -> encoder_mu
        self.enc_mu = nn.GRUCell(3*self.patch_size, self.enc_hidden_size)
        #writer -> encoder_logsigma
        self.enc_logvar = nn.GRUCell(3*self.patch_size, self.enc_hidden_size)
        #hidden_mu->mu 
        self.mu_fc = nn.Linear(self.enc_hidden_size, self.z_size)
        #hidden_logvar->logvar
        self.logvar_fc = nn.Linear(self.enc_hidden_size, self.z_size)
        #do z = mu+\epsilon sigma in reparametrize
        #send to decoder hidden RNN 

        self.dec_rnn = nn.GRUCell(self.z_size, self.dec_hidden_size)
        self.write   = nn.Linear(self.dec_hidden_size, self.input_size)
        self.get_attn_params = nn.Linear(dec_hidden_size, 5)
        self.write_patch = nn.Linear(dec_hidden_size, patch_size)

        #self.enc_input = Variable(torch.zeros(self.seq_len, self.batch_size, 3*self.input_size))
        
    def read(self, x, x_hat):
        return torch.cat((x, x_hat), -1)

    def read_attn(self, x, x_hat, F_x, F_y, gamma):
        #implements F_y x F_x_T [F_y times x times F_x_transpose]
        F_x_t = torch.t(F_x)
        tmp_x = torch.mm(x, F_x_t)
        tmp_x = torch.mm(F_y, tmp_x)
        
        tmp_x_hat = torch.mm(x_hat, F_x_t)
        tmp_x_hat = torch.mm(F_y, tmp_x_hat)

        #this should have size 2*NxN == 2*patch_size 
        return torch.cat(tmp_x, tmp_x_hat)

    def write_attn(self, h_dec, F_x, F_y, gamma):
        #implements F_y_T w F_x 
        # w is output patch of size NxN
        w = self.write_patch(h_dec)
        F_y_t = torch.t(F_y)
        tmp = torch.mm(w, F_x)
        tmp = 1/self.gamma*torch.mm(F_y_t, tmp)
        #tmp is of size BXA
        return tmp
        

    def encoder_RNN(self, r, h_mu_prev, h_logvar_prev, h_dec_prev, seq_id):
        enc_input = torch.cat((r, h_dec_prev), -1) #skip connection from decoder 

        h_mu = self.enc_mu(enc_input, h_mu_prev)
        mu = F.relu(self.mu_fc(h_mu))
        h_logvar = self.enc_logvar(enc_input, h_logvar_prev)
        logvar = F.relu(self.logvar_fc(h_logvar))

        #print("encoder done")
        #print("------------")

        return mu, h_mu, logvar, h_logvar
 

    def decoder_network(self, z, h_dec_prev, c):
        h_dec= self.dec_rnn(z, h_dec_prev)
        c = c + self.write(h_dec)
        #print("decoder done")
        #print("------------")

        return c, h_dec

    def decoder_network_attn(self, z, h_dec_prev, c):
        A = self.A
        B = self.B
        N = self.N
        
        h_dec = self.dec_rnn(z, h_dec_prev)
        #use decoder to get attention parameters 
        g_x, g_y, logvar, logdelta, loggamma = self.get_attn_params(self.h_dec)
        
        delta = logdelta.exp_()
        gamma = loggamma.exp_()
        var = logvar.exp_()

        g_x = (A+1)*(g_x+1)/2
        g_y = (B+1)*(g_y+1)/2
        delta = (max(A,B)-1)/(N-1)*delta 

        F_x, F_y = compute_filterbank_matrices(g_x, g_y, var, delta, N)

        #c_t is of shape NxN
        c_t = c_t + self.write_attn(gamma, F_x, F_y, h_dec)
        return c_t, h_dec
        
    def reparametrize_and_sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 

        #sampling epsilon
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        
        #mu + sigma * epsilon
        #print("reparametrize and sample done")
        #print("-----------------------------")
        return eps.mul(std).add_(mu)

    def forward(self, x_in, T):
        #advance by T timesteps 
        x = x_in.view(-1, input_size).cuda() #flatten
        c = Variable(torch.zeros(self.batch_size, self.input_size)).cuda()
        h_mu = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        h_logvar = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        mu = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        logvar = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        h_dec = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()

        mu_t = []
        logvar_t = []
            
        for seq in range(T):
            x_hat = x - F.sigmoid(c)
            r = self.read(x, x_hat) #cat operation
            mu, h_mu, logvar, h_logvar = self.encoder_RNN(r, h_mu, h_logvar, h_dec, seq)
            z = self.reparametrize_and_sample(mu, logvar)
            c, h_dec = self.decoder_network(z, h_dec, c)
            mu_t.append(mu)
            logvar_t.append(logvar)


        #print("FORWARD PASS DONE")
        #print("=================")

        return F.sigmoid(c), mu_t, logvar_t


input_size = 784
seq_len = 20
batch_size = args.batch_size #128
model = draw(input_size, seq_len, batch_size)
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
 
def loss_function(recon_x, x, mu, logvar, T):
    #after T timesteps, we compare reconstruction with original
    BCE = reconstruction_function(recon_x, x)
    #KLD loss
    #1/2*(mu^2 + sigma^2 - log sigma^2)
    #(mu^2+ sigma^2)*-1 + log sigma^2 
    #    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    #=-1/2*KLD
    #KLD = torch.sum(KLD_element).mul_(-0.5)

    # we modify this for DRAW 
    # = sum over time (sum over batch KLD) -T/2

    BCE = reconstruction_function(recon_x, x)

    KLD = 0.0

    for seq in range(T):
        KLD_element = mu[seq].pow(2).add_(logvar[seq].exp()).mul_(-1).add_(logvar[seq])
        KLD += torch.sum(KLD_element).mul_(-0.5)
        
    KLD += -T*0.5

    return BCE + KLD 

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch, T):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        h_data = Variable(data)
        #print('data.size()', data.size())
        data = h_data.squeeze(0)
        #print('squeezed data.size()', data.size())
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu_t, logvar_t = model(data, T)
        loss = loss_function(recon_batch, data, mu_t, logvar_t, T)
        loss.backward()

        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))



            #download to host             
            samples = recon_batch.data.cpu().numpy()[:16]
            
            fig = plt.figure(figsize=(4,4))
            gs  = gridspec.GridSpec(4,4)
            gs.update(wspace=0.05, hspace=0.05)


            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28,28), cmap='gray')

            if not os.path.exists('out/'):
                os.makedirs('out/')
            
            plt.savefig('out/snapshot.png', bbox_inches='tight')
            plt.close(fig)

            samples = data.data.cpu().numpy()[:16]
            fig = plt.figure(figsize=(4,4))
            gs  = gridspec.GridSpec(4,4)
            gs.update(wspace=0.05, hspace=0.05)


            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28,28), cmap='gray')

            if not os.path.exists('out/'):
                os.makedirs('out/')
            
            plt.savefig('out/snapshot_o.png', bbox_inches='tight')
            plt.close(fig)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



for epoch in range(10):
    train(epoch, seq_len)
    
        
        
