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
        self.patch_size = patch_size 
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

        self.dec_rnn = nn.LSTMCell(self.z_size, self.dec_hidden_size)
        self.write   = nn.Linear(self.dec_hidden_size, self.input_size)
        self.attn_params = nn.Linear(self.dec_hidden_size, 5)
        self.write_patch = nn.Linear(self.dec_hidden_size, patch_size)

    def read(self, x, x_hat):
        return torch.cat((x, x_hat), -1)

    def read_attn(self, x, x_hat, F_x, F_y, gamma):
        #implements F_y x F_x_T [F_y times x times F_x_transpose]
        #to get patch vector of size N X N
        #x and x_hat are of size BxA

        x = x.view(-1, self.B, self.A)
        x_hat = x_hat.view(-1, self.B, self.A)

        tmp_x = Variable(torch.zeros(self.batch_size, self.N, self.N)).cuda()
        tmp_x_hat = Variable(torch.zeros(self.batch_size, self.N, self.N)).cuda()

        batch_size = self.batch_size
        N = self.N
        B = self.B
        A = self.A 

        #another batch opertion
        # v = gamma . F_y . x . F_x_t
        # F_x: bsz X N X A
        # F_x_t: bsz X A X N 
        # F_y: bsz X B X N
        # gamma: bsz

        #x: B X A 
        #x_hat: B X A
        

        #for i in range(batch_size):
        #    F_x_t = torch.t(F_x[i])
        #    tmp_x[i] = torch.mm(F_y[i], torch.mm(x[i], F_x_t))*gamma[i]
        #    tmp_x_hat[i] = torch.mm(F_y[i], torch.mm(x_hat[i], F_x_t))*gamma[i]

        F_x_t = F_x.permute(0, 2, 1)
        tmp_x = gamma.expand(N, N, batch_size).permute(0, 1, 2) * F_y.bmm(x.bmm(F_x_t))
        tmp_x_hat = gamma.expand(N, N, batch_size).permute(0, 1, 2) * F_y.bmm(x_hat.bmm(F_x_t))
        
        

        print("Read done")
        print("=========")

        #this should have size 2*NxN == 2*patch_size 
        return torch.cat((tmp_x.view(-1, self.patch_size), tmp_x_hat.view(-1, self.patch_size)), -1)

    def write_attn(self, h_dec, F_x, F_y, gamma):
        #implements F_y_T w F_x 
        # w is output patch of size NxN

        #F_x is of size [batch_size x N x A]
        #F_y is of size [batch_size x N x B]


        batch_size = self.batch_size
        w = self.write_patch(h_dec).view(-1, self.N, self.N)

        #w: bsz, N x N
        #F_x: bsz, N x A
        #tmp = Variable(torch.zeros(batch_size, B, A)).cuda()

        #for batch in range(batch_size):
        #    F_y_t = torch.t(F_y[batch])
        #    tmp1 = torch.mm(w[batch], F_x[batch])
        #    tmp[batch] = 1.0/gamma[batch]*torch.mm(F_y_t, tmp1)
        #tmp is of size BXA
        
        #F_y=> bsz X N X B
        #F_y_t => bsz X B x N
        #w => bsz X N X N
        #F_x => bsz X N X A
        # gamma => bsz 

        # F_y_t . w . F_x => bsz X B X A
        #1/gamma * F_Y_t . w . F_x => bsz X B X A

        F_y_t = F_y.permute(0,2,1) 
        tmp = F_y_t.bmm(w.bmm(F_x))

        epsilon = 0.0001*Variable(torch.ones(batch_size).cuda())

        g = (gamma+epsilon).expand(B,A,batch_size).permute(2,0,1)
        tmp = 1.0/g * tmp

        return tmp
        

    def encoder_RNN(self, r, h_mu_prev, h_logvar_prev, h_dec_prev, seq_id):
        enc_input = torch.cat((r, h_dec_prev), -1) #skip connection from decoder 

        h_mu = self.enc_mu(enc_input, h_mu_prev)
        mu = F.relu(self.mu_fc(h_mu))
        h_logvar = self.enc_logvar(enc_input, h_logvar_prev)
        logvar = F.tanh(self.logvar_fc(h_logvar))

        print("encoder done")
        print("------------")

        return mu, h_mu, logvar, h_logvar
 

    def decoder_network(self, z, h_dec_prev, c):
        h_dec= self.dec_rnn(z, h_dec_prev)
        c = c + self.write(h_dec)
        #print("decoder done")
        #print("------------")

        return c, h_dec

    def get_attn_params(self, h_dec):
        params = self.attn_params(h_dec)
        g_x = params[:,0]
        g_y = params[:,1]
        logvar = params[:,2]
        logdelta = params[:,3]
        loggamma = params[:,4]

        return g_x, g_y, logvar, logdelta, loggamma
        
        

    def decoder_network_attn(self, z, h_dec_prev, c_dec_prev, c_t, F_x, F_y):
        A = self.A
        B = self.B
        N = self.N
        
        h_dec, c_dec = self.dec_rnn(z, (h_dec_prev, c_dec_prev))
        #use decoder to get attention parameters 
        g_x, g_y, logvar, logdelta, loggamma = self.get_attn_params(h_dec)
        
        delta = torch.exp(logdelta)
        gamma = torch.exp(loggamma)
        var = torch.exp(logvar)

        g_x = (A+1)*(g_x+1)/2
        g_y = (B+1)*(g_y+1)/2
        delta = (max(A,B)-1)/(N-1)*delta 

        F_x, F_y = self.compute_filterbank_matrices(g_x, g_y, delta, var, F_x, F_y, N, A, B, self.batch_size)
        #        F_x[:,:,:] = 1


        print("Computed FB matrices")

        #c_t is of shape NxN
        c_t = c_t + self.write_attn(h_dec, F_x, F_y, gamma)
        #c_t = c_t + self.write(h_dec)


        print("decoder_network_attn done")
        print("=========================")
        return c_t , h_dec, c_dec, F_x, F_y, gamma

    def compute_filterbank_matrices(self, g_x, g_y, delta, var, F_x, F_y, N, A, B, batch_size):
        #F_x = Variable(torch.ones(self.batch_size, self.N, self.A)).cuda()
        #F_y = Variable(torch.ones(self.batch_size, self.N, self.B)).cuda()

        i = torch.arange(0, N).cuda()
        
        gx  = g_x.expand(N, A, batch_size)
        gx  = gx.permute(2,0,1)

        i = Variable(i.expand(batch_size, A, N).permute(0,2,1))
        #i = Variable(i.permute(0, 2, 1))

        dx = delta.expand(N, A, batch_size).permute(2, 0, 1)

        mu_i = gx + i*dx - (N/2 + 0.5) * dx

        a = torch.arange(0,A).cuda()
        a = Variable(a.expand(batch_size, N, A))

        vx = var.expand(N, A, batch_size)
        vx = vx.permute(2, 0, 1)

        F_x = torch.exp(-(a-mu_i)*(a-mu_i)/(2.0*vx))

        n_x = torch.sum(F_x, 2).expand(A, batch_size, N)
        n_x = n_x.permute(1, 2, 0)

        F_x = F_x/n_x

        #now compute F_y

        gy = g_y.expand(N, A, batch_size).permute(2, 0, 1)
        dy = delta.expand(N, B, batch_size).permute(2, 0, 1)

        j = torch.arange(0, N).cuda()
        j = Variable(j.expand(batch_size, B, N)).permute(0, 2, 1)

        mu_j = gy + j*dy - (N/2 + 0.5) * dy

        b = torch.arange(0, B).cuda()
        b = Variable(b.expand(batch_size, N, A))

        vy = var.expand(N, B, batch_size)
        vy = vy.permute(2, 0, 1)

        F_y = torch.exp(-(b-mu_j)*(b-mu_j)/(2.0*vy))

        n_y = torch.sum(F_x, 2).expand(A, batch_size, N)
        n_y = n_y.permute(1, 2, 0)

        F_y = F_y/n_y
        
        
        return F_x, F_y

    def reparametrize_and_sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 

        #sampling epsilon
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        
        #mu + sigma * epsilon
        print("reparametrize and sample done")
        #print("-----------------------------")
        return eps.mul(std).add_(mu)

    def forward(self, x_in, T):
        #advance by T timesteps 
        x = x_in.view(-1, input_size).cuda() #flatten
        #        c = Variable(torch.randn(self.batch_size, self.input_size)).cuda()
        c = Variable(torch.zeros(self.batch_size, self.input_size)).cuda()
        h_mu = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        h_logvar = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        mu = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        logvar = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        h_dec = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        c_dec = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
        
        #filterbank variables 
        F_x = Variable(torch.ones(self.batch_size, self.N, self.A), requires_grad=False).cuda()
        F_y = Variable(torch.ones(self.batch_size, self.N, self.B), requires_grad=False).cuda()
        gamma = Variable(torch.ones(self.batch_size), requires_grad=False).cuda()

        mu_t = []
        logvar_t = []

        print("Starting forward")

        for seq in range(T):
            x_hat = x - F.sigmoid(c)
            r = self.read_attn(x, x_hat, F_x, F_y, gamma)
            mu, h_mu, logvar, h_logvar = self.encoder_RNN(r, h_mu, h_logvar, h_dec, seq)
            z = self.reparametrize_and_sample(mu, logvar)
            c, h_dec, c_dec, F_x, F_y, gamma = self.decoder_network_attn(z, h_dec, c_dec, c, F_x, F_y)

            mu_t.append(mu)
            logvar_t.append(logvar)
            print('seqnorm', torch.norm(c))
            print('seq done')
            print('--------')


        #print("FORWARD PASS DONE")
        #print("=================")

        return F.sigmoid(c), mu_t, logvar_t

A = 28
B = 28
N = 12
input_size = A * B #=784
patch_size = N * N #=144
seq_len = 20
batch_size = args.batch_size #100
model = draw(input_size, patch_size, A, B, N,  seq_len, batch_size)
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
#reconstruction_function = nn.MSELoss()
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

    print(model)
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
        
        print("Doing Backprop")
        print("==============")
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
    
        
        
