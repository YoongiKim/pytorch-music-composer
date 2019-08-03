import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import model
from model import log, log_empty
from dataset import MidiDataset
import os
import nonechucks as nc
from torch_logger import TorchLogger
import numpy as np

def save_sample_images(images, step, name='real_images', filter=True):
    if filter:
        images = images > 0.5
    images = images.reshape(-1, 1, num_samples * height, width)
    save_image(images[0:1], os.path.join(output_dir, '{}-{}-{}.png'.format(name, epoch, step)))

def save_sample_music(images, epoch, step):
    import numpy_to_midi
    images = images.reshape(-1, num_samples, height, width)
    npy_file = os.path.join(output_dir, 'sample-{}-{}.npy'.format(epoch, step))
    np.save(npy_file, images.detach().cpu().numpy()[0])
    numpy_to_midi.numpy_to_midi(npy_file, npy_file + '.mid', ticks_per_beat=48)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    height = 96
    width = 96
    num_samples = 8
    num_epochs = 200
    batch_size = 4
    output_dir = 'output'
    h_dim=2048
    z_dim=128

    os.makedirs(output_dir, exist_ok=True)

    dataset = nc.SafeDataset(MidiDataset(npy_glob_pattern='vgmusic_npy_point/Nintendo 08 DS/**/*.npy', num_samples=num_samples))
    dataloader = nc.SafeDataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = model.VAE(num_samples, height, width, h_dim, z_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_step = len(dataset) // batch_size
    logger = TorchLogger(num_epochs, total_step)

    for epoch in range(num_epochs):
        logger.reset()

        for i, x in enumerate(dataloader):
            x = x.to(device)

            x_reconst, mu, log_var = model(x)

            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            logger.log(epoch=epoch+1, step=i+1, reconst_loss=reconst_loss.item(), kl_div=kl_div.item())

        with torch.no_grad():
            z = torch.randn(batch_size, z_dim).to(device)
            out = model.decode(z)
            save_sample_images(out, i+1, 'sampled', filter=False)

            out = model(x)
            save_sample_images(x, i+1, 'real', filter=True)
            save_sample_images(x_reconst, i + 1, 'reconst', filter=False)

        # torch.save(model, os.path.join(output_dir, 'modelVAE-{}'.format(epoch + 1)))
