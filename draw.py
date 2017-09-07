from __future__ import print_function
import argparse
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

class draw(nn.module):
    def __init__(self, input_size, seq_len = 1,  num_layers = 1):
        super(VAE, self).__init__()
        #self.read()
        #self.encoder_RNN()
        #self.Q() #mu, sigma, includes reparametrization etc.
        #self.sample()
        #self.decoder_RNN()
        #self.write()

        dec_hidden_size = input_size
        enc_hidden_size = input_size
        hidden_size = input_size 
        z_size = hidden_size/2
        #        self.enc_mu = nn.gru(input_size, enc_hidden_size, num_layers)
        #        self.enc_logsigma = nn.gru(input_size, enc_hidden_size, num_layers)
        #        self.write = nn.gru(dec_hidden_size, input_size, num_layers)

        #writer -> encoder_mu
        self.enc_mu = nn.GRUCell(2*input_size, enc_hidden_size)
        #writer -> encoder_logsigma
        self.enc_logsigma = nn.GRUCell(2*input_size, enc_hidden_size)
        #hidden_mu->mu 
        self.mu_fc = nn.Linear(enc_hidden_size, z_size)
        #hidden_logsigma->logsigma
        self.logsigma_fc = nn.Linear(enc_hidden_size, z_size)
        #do z = mu+\epsilon sigma in reparametrize
        #send to decoder hidden RNN 

        self.dec_rnn = nn.GRUCell(z_size, dec_hidden_size)
        self.write   = nn.Linear(dec_hidden_size, input_size)
        

        def read(x, x_hat):
            return torch.cat((x, x_hat), 0)

        def encoder_RNN(r, seq_id):
            #2 RNNs, one for mu, one for logsigma 
            '''
            #formulas from paper without attention
            enc_input = torch.cat(r, h_dec) #skip connection from decoder
            mu, self.h_mu = self.enc_mu(x, self.h_mu)
            self.mu = nn.ReLU(mu)
            logsigma, self.h_logsigma = self.enc_logsigma(x, self.h_logsigma)
            logsigma = nn.tanh(logsigma)
            return mu, logsigma'''

            if(seq_id==0):
                h_mu_prev = torch.zeros(self.h_mu.size(1), self.h_mu.size(2))
                h_logsigma_prev = torch.zeros(self.h_logsigma.size(1), self.h_logsigma.size(2))
                h_dec = torch.zeros(h_dec.size(1), h_dec.size(2))
            else:
                h_mu_prev = self.h_mu[seq_id-1]
                h_logsigma_prev = self.h_logsigma[seq_id-1]
                h_dec_prev = self.h_dec[seq_id-1]
            
            enc_input = torch.cat(r, h_dec_prev) #skip connection from decoder 

            self.h_mu[seq_id] = self.enc_mu(enc_input, h_mu_prev)
            mu = nn.ReLu(self.mu_fc(self.h_mu[seq_id]))

            self.h_logsigma[seq_id] = self.enc_sigma(enc_input, h_logsigma_prev)
            logsigma = nn.ReLu(self.sigma_fc(self.h_logsigma[seq_id]))
            
            return mu, logsigma

        def decoder_network(self, z, seq_id):
            if(seq_id==0):
                h_dec_prev = torch.zeros(self.decoder_hidden_size)
            else:
                h_dec_prev = h_dec[seq_id-1]

            self.h_dec[seq_id] = dec_rnn(z, h_dec_prev)
            c = nn.Sigmoid(write(self.h_dec[seq_id]))
            return c
            
            
        def reparametrize_and_sample(self, mu, logsigma):
            std = logsigma.mul(0.5).exp_()

            if args.cuda:
                eps = torch.cuda.FloatTensor(std.size()).normal_()
            else:
                eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
            return eps.mul(std).add_(mu)

        def forward(self, ....):
            #advance by T timesteps 
            mu_t = []
            logsigma_t = []
            x_t = []
            for seq in range(T):
                x_hat = x - nn.Sigmoid(self.c)
                r = self.read(x, x_hat)
                mu, logsigma = self.encoder_RNN(r, seq)
                z = self.reparametrize(mu, logsigma)
                c = self.decoder_network(z, seq)
                mu_t.append(mu)
                logsigma_t.append(mu)
                x_t.append(c)
            
            return mu_t, logsigma_t, x_t

            
            
model = draw()
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

    KLD = 0.0f

    for seq in range(T):
        KLD_element = mu[seq].pow(2).add_(logvar[seq].exp()).mul_(-1).add_(logvar[seq])
        KLD += torch.sum(KLD_element).mul_(-0.5)
        
    KLD += -T*0.5

    return BCE + KLD 




    
            
            


    
        
        
