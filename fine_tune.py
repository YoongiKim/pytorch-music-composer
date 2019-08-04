import torch
import numpy as np
from modules import model
from model import TimeDistributedLinear
import torch.nn as nn
#%%
model = model.VAE(n=8, h=96, w=96, h_dim=2048, z_dim=128)
model.load_state_dict(torch.load('params.ckpt'))
#%%
model.fc2 = nn.Linear(2048, 20)
model.fc3 = nn.Linear(2048, 20)
model.fc5 = TimeDistributedLinear(2048, h*w, bn=False, activation=nn.Sigmoid)