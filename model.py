"""
samples [batch, samples, 96, 96]

GAN
Z -> output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def log(tensor):
    if type(tensor) is tuple:
        msg = ""
        for t in tensor:
            msg += '{} '.format(list(t.size()))
        print(msg)

    else:
        print('{}'.format(list(tensor.size())))
    return tensor

def log_empty(tensor):
    return tensor


"""
G
[2, 200]
[2, 4096]
[2, 4096]
[2, 4096]
[2, 16, 256]
[2, 16, 9216]
[2, 147456]
[2, 147456]
[2, 16, 96, 96]
D
[2, 16, 96, 96]
[2, 16, 9216]
[2, 16, 256]
[2, 4096]
[2, 4096]
[2, 4096]
[2, 1]
"""


class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, num_samples, h, w):
        super(Generator, self).__init__()
        self.hidden_size, self.num_samples, self.h, self.w = hidden_size, num_samples, h, w

        self.l1 = nn.Linear(latent_size, num_samples * hidden_size)
        self.b1 = nn.BatchNorm1d(num_samples * hidden_size)
        self.a1 = nn.LeakyReLU()

        self.l2 = nn.Linear(hidden_size, h * w)
        self.a2 = nn.LeakyReLU()

    def forward(self, x, log):
        log(x)
        x = log(self.l1(x))
        x = log(self.b1(x))
        x = log(self.a1(x))
        x = log(x.view(-1, self.num_samples,  self.hidden_size))

        x = log(self.l2(x))
        x = log(x.view(-1, self.num_samples * self.h * self.w))
        x = log(self.a2(x))
        x = log(x.view(-1, self.num_samples, self.h, self.w))

        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_samples, h, w):
        super(Discriminator, self).__init__()
        self.hidden_size, self.num_samples, self.h, self.w = hidden_size, num_samples, h, w

        self.l1 = nn.Linear(h * w, hidden_size)
        self.b1 = nn.BatchNorm1d(num_samples * hidden_size)
        self.a1 = nn.LeakyReLU()

        self.l2 = nn.Linear(num_samples * hidden_size, 1)

    def forward(self, x, log):
        log(x)
        x = log(x.view(-1, self.num_samples, self.h * self.w))
        x = log(self.l1(x))
        x = log(x.view(-1, self.num_samples * self.hidden_size))
        x = log(self.b1(x))
        x = log(self.a1(x))
        x = log(self.l2(x))

        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator(200, 256, 16, 96, 96)
    D = Discriminator(256, 16, 96, 96)
    input = torch.randn(2, 200)
    print('G')
    output = G(input, log)
    print('D')
    output = D(output, log)
