from __future__ import print_function
import argparse
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = args.no_cuda


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
    def __init__(self, input_size, seq_len = 1):
        super(draw, self).__init__()
        #self.read()
        #self.encoder_RNN()
        #self.Q() #mu, sigma, includes reparametrization etc.
        #self.sample()
        #self.decoder_RNN()
        #self.write()

        self.input_size = input_size 
        self.dec_hidden_size = input_size
        self.enc_hidden_size = input_size
        self.hidden_size = input_size 
        self.z_size = int(self.hidden_size/2)

        #        self.enc_mu = nn.gru(input_size, enc_hidden_size, num_layers)
        #        self.enc_logsigma = nn.gru(input_size, enc_hidden_size, num_layers)
        #        self.write = nn.gru(dec_hidden_size, input_size, num_layers)

        #writer -> encoder_mu
        self.enc_mu = nn.GRUCell(2*self.input_size, self.enc_hidden_size)
        #writer -> encoder_logsigma
        self.enc_logsigma = nn.GRUCell(2*self.input_size, self.enc_hidden_size)
        #hidden_mu->mu 
        self.mu_fc = nn.Linear(self.enc_hidden_size, self.z_size)
        #hidden_logsigma->logsigma
        self.logsigma_fc = nn.Linear(self.enc_hidden_size, self.z_size)
        #do z = mu+\epsilon sigma in reparametrize
        #send to decoder hidden RNN 

        self.dec_rnn = nn.GRUCell(self.z_size, self.dec_hidden_size)
        self.write   = nn.Linear(self.dec_hidden_size, self.input_size)
        

        def read(x, x_hat):
            return torch.cat((x, x_hat), -1)

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
            
            enc_input = torch.cat((r, h_dec_prev), -1) #skip connection from decoder 

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

        def forward(self, x, T):
            #advance by T timesteps 
            mu_t = []
            logsigma_t = []

            x_t_prev = Variable(torch.zeros(x.size(1), x.size(2)))
            
            for seq in range(T):
                x_hat = x - nn.Sigmoid(x_t_prev)
                r = self.read(x, x_hat) #cat operation
                mu, logsigma = self.encoder_RNN(r, seq) 
                z = self.reparametrize(mu, logsigma)
                c = self.decoder_network(z, seq)
                mu_t.append(mu)
                logsigma_t.append(logsigma)
                x_t_prev = c
            
            return c, mu_t, logsigma_t


input_size = 784
seq_len = 2
model = draw(input_size, seq_len)
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
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        #mu_t, logsigma_t are arrays
        recon_batch, mu_t, logsigma_t = model(data, T)

        loss = loss_function(recon_batch, data, mu_t, logsigma_t, T)


for epoch in range(1):
    train(epoch, seq_len)
    
        
        
