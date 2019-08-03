import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def log(tensor):
    print('{}'.format(list(tensor.size())))
    return tensor

def log_empty(tensor):
    return tensor

def log_img(tensor):
    to_pil = torchvision.transforms.ToPILImage()
    img = tensor.clone().detach()

    if len(list(img.size())) == 4:
        img = img.squeeze(0)

    img = to_pil(img)
    plt.imshow(img)
    plt.title(list(tensor.size()))
    plt.show()
    return tensor

"""
D
[2, 16, 96, 96]
[2, 16, 9216]
[2, 16, 256]
[2, 4096]
[2, 2048]
G
[2, 128]
[2, 4096]
[2, 16, 256]
[2, 16, 2048]
[2, 16, 9216]
[2, 16, 96, 96]
"""


class TimeDistributedLinear(nn.Module):
    """
    [Batch, Seq, Word] -> [Batch * Seq, Word] -> [Batch, Seq, Word]
    """
    def __init__(self, in_features, out_features, bn=True, activation=nn.LeakyReLU):
        super(TimeDistributedLinear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bn, self.activation = bn, activation
        self.fc = nn.Linear(in_features, out_features)
        if bn:
            self.bn = nn.BatchNorm1d(out_features)
        if activation is not None:
            self.act = activation()

    def forward(self, x):
        input_shape = x.size()

        assert len(input_shape) == 3, "expected dim=3"

        bs = x.size(0)
        seq_len = x.size(1)
        word_len = x.size(2)

        assert word_len == self.in_features, "expected input_shape[-1] == in_features"

        x = x.view(-1, word_len)
        x = self.fc(x)

        if self.bn:
            x = self.bn(x)
        if self.activation is not None:
            x = self.act(x)

        x = x.view(bs, seq_len, self.out_features)

        return x


class LinearBlock(nn.Module):
    """
    [Batch, Seq, Word] -> [Batch * Seq, Word] -> [Batch, Seq, Word]
    """
    def __init__(self, in_features, out_features, bn=True, activation=nn.LeakyReLU):
        super(LinearBlock, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bn, self.activation = bn, activation
        self.fc = nn.Linear(in_features, out_features)
        if bn:
            self.bn = nn.BatchNorm1d(out_features)
        if activation is not None:
            self.act = activation()

    def forward(self, x):
        x = self.fc(x)

        if self.bn:
            x = self.bn(x)
        if self.activation is not None:
            x = self.act(x)

        return x


class Encoder(nn.Module):
    def __init__(self, n=16, h=96, w=96):
        super(Encoder, self).__init__()
        self.n, self.h, self.w = n, h, w
        self.encoder1 = nn.Sequential(TimeDistributedLinear(96*96, 2048),
                                      TimeDistributedLinear(2048, 256))
        self.encoder2 = nn.Sequential(LinearBlock(256 * n, 2048))

    def forward(self, x, log=log_empty):
        log(x)  # [2, 16, 96, 96]
        x = log(x.view(-1, self.n, self.h*self.w))
        x = log(self.encoder1(x))
        x = log(x.view(-1, self.n * x.size(-1)))
        x = log(self.encoder2(x))

        return x

class Decoder(nn.Module):
    def __init__(self, n=16, h=96, w=96, z_dim=128):
        super(Decoder, self).__init__()
        self.n, self.h, self.w = n, h, w
        self.decoder1 = nn.Sequential(LinearBlock(z_dim, 2048),
                                      LinearBlock(2048, 256*n))
        self.decoder2 = nn.Sequential(TimeDistributedLinear(256, 2048))

    def forward(self, x, log=log_empty):
        log(x)  # [2, 128]
        x = log(self.decoder1(x))  # [2, 4096]
        x = log(x.view(-1, self.n, x.size(-1)//self.n))  # [2, 16, 256]
        x = log(self.decoder2(x))  # [2, 16, 2048]

        return x

class Discriminator(nn.Module):
    def __init__(self, n=16, h=96, w=96):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(n, h, w)
        self.fc = LinearBlock(2048, 128, activation=nn.Sigmoid)

    def forward(self, x, log=log_empty):
        x = self.encoder(x, log)
        x = self.fc(x)
        return x

class Generator(nn.Module):
    def __init__(self, n=16, h=96, w=96):
        super(Generator, self).__init__()
        self.n, self.h, self.w = n, h, w
        self.decoder = Decoder(n, h, w)
        self.fc = TimeDistributedLinear(2048, self.h*self.w, activation=nn.Tanh)

    def forward(self, x, log=log_empty):
        x = self.decoder(x, log)
        x = log(self.fc(x))
        x = log(x.view(-1, self.n, self.h, self.w))

        return x


"""
Encoder
[2, 16, 96, 96]
[2, 16, 9216]
[2, 16, 256]
[2, 4096]
[2, 2048]
Decoder
[2, 128]
[2, 4096]
[2, 16, 256]
[2, 16, 2048]
"""

class VAE(nn.Module):
    def __init__(self, n=16, h=96, w=96, h_dim=2048, z_dim=128):
        super(VAE, self).__init__()
        self.n, self.h, self.w = n, h, w
        self.fc1 = Encoder(n, h, w)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = Decoder(n, h, w, z_dim=z_dim)
        self.fc5 = TimeDistributedLinear(h_dim, h*w, bn=False, activation=nn.Sigmoid)

    def encode(self, x):
        h = self.fc1(x)
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc4(z)
        h = self.fc5(h)
        h = h.view(-1, self.n, self.h, self.w)
        return h

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


if __name__ == "__main__":
    device = torch.device('cpu')
    # En = Encoder().to(device)
    # De = Decoder().to(device)
    # D = Discriminator().to(device)
    # G = Generator().to(device)
    vae = VAE().to(device)

    # print('Encoder')
    # out = En(torch.randn(2, 16, 96, 96), log)
    # print('Decoder')
    # out = De(torch.randn(2, 128), log)

    print('VAE')
    out = vae(torch.randn(2, 16, 96, 96))
