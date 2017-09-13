from __future__ import print_function
import argparse
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


def compute_filterbank_matrices(g_x, g_y, delta, sigma, N, A, B):
    F_x = Variable(torch.zeros(N, A)).cuda()
    F_y = Variable(torch.zeros(N, B)).cuda()

    for i in range(N):
        mu_i = g_x + (i- N/2 - 0.5)*delta
        Z_x = 0
        for a in range(A):
            F_x[i,a] = np.exp(-(a-mu_i)*(a-mu_i)/(2.0*sigma*sigma))
            Z_x += F_x[i,a]
        F_x[i,:] = F_x[i, :]/Z_x

    for j in range(N):
        mu_j = g_y + (j-N/2 -0.5)*delta
        Z_y = 0
        for j in range(B):
            F_y[j, b] = np.exp(-(b-mu_j)*(b-mu_j)/(2.0*sigma*sigma))
            Z_y += F_y[j, b]
        F_y[j,:] = F_y[j, :]/Z_y

    return F_x, F_y 

