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

class Generator(nn.Module):
    def __init__(self, input_shape=(768, 96), output_shape=(768, 96)):
        super(Generator, self).__init__()
        self.input_shape, self.output_shape = input_shape, output_shape

        self.dropout = nn.Dropout2d(p=0.1)
        self.lstm = nn.LSTM(96, 512, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
                nn.Linear(512*2, output_shape[1]))


    def forward(self, x, log=log_empty):
        x = log(x.view(-1, 1, self.input_shape[0], self.input_shape[1]))
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        lstm = log(F.leaky_relu(self.lstm(x.squeeze(1))[0]))
        out = log(self.fc(lstm))

        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape=(768, 96), output_shape=(1, 1)):
        super(Discriminator, self).__init__()
        self.input_shape, self.output_shape = input_shape, output_shape

        self.lstm = nn.LSTM(96, 96, 1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(96 * 2, 1)
        self.fc2 = nn.Sequential(
            nn.Linear(input_shape[0], 1)
        )

    def forward(self, x, log=log_empty):
        x = log(x.view(-1, 1, self.input_shape[0], self.input_shape[1]))
        self.lstm.flatten_parameters()
        lstm = log(F.leaky_relu(self.lstm(x.squeeze(1))[0]))
        out = log(F.leaky_relu(self.fc1(lstm)))
        out = log(out.squeeze(2))
        out = log(self.fc2(out))

        return out


if __name__ == "__main__":
    device = torch.device('cpu')
    G = Generator().to(device)
    D = Discriminator().to(device)

    out = G(torch.randn(1, 768, 96), log)
    print('D')
    D(out, log)
