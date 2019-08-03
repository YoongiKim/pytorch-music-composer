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

def sample_log(real_images, fake_images):
    real_images = real_images.reshape(-1, 1, num_samples * height, width)
    save_image(real_images[0:1], os.path.join(output_dir, 'real_images.png'))

    fake_images = fake_images > 0.5
    fake_images = fake_images.reshape(-1, 1, num_samples * height, width)
    save_image(fake_images[0:1], os.path.join(output_dir, 'fake_images-{}-{}.png'.format(epoch + 1, i + 1)))

    import numpy_to_midi
    fake_images = fake_images.reshape(-1, num_samples, height, width)
    npy_file = os.path.join(output_dir, 'sample-{}-{}.npy'.format(epoch + 1, i + 1))
    np.save(npy_file, fake_images.detach().cpu().numpy()[0])
    numpy_to_midi.numpy_to_midi(npy_file, npy_file + '.mid', ticks_per_beat=48)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    height = 96
    width = 96
    num_samples = 8
    num_epochs = 200
    batch_size = 16
    output_dir = 'output'

    os.makedirs(output_dir, exist_ok=True)

    dataset = nc.SafeDataset(MidiDataset(npy_glob_pattern='vgmusic_npy/Nintendo 08 DS/**/*.npy', num_samples=num_samples))
    dataloader = nc.SafeDataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    G = model.Generator(input_shape=(height*num_samples, width), output_shape=(height*num_samples, width)).to(device)
    D = model.Discriminator(input_shape=(height*num_samples, width), output_shape=(1)).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=0.0002)
    d_opt = torch.optim.Adam(D.parameters(), lr=0.0002)
    criterion = torch.nn.MSELoss()

    total_step = len(dataset) // batch_size
    logger = TorchLogger(num_epochs, total_step)

    for epoch in range(num_epochs):
        logger.reset()

        for i, images in enumerate(dataloader):
            images = images.to(device)
            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            # z = torch.randn_like(images).to(device)
            z = images
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # Backprop and optimize
            d_loss = d_loss_real + d_loss_fake
            d_opt.zero_grad()
            g_opt.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 0.001)
            d_opt.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            # z = torch.randn_like(images).to(device)
            z = images
            fake_images = G(z)
            outputs = D(fake_images)

            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels) + criterion(fake_images, images.view(-1,1, num_samples*height, width))

            # Backprop and optimize
            d_opt.zero_grad()
            g_opt.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 0.001)
            g_opt.step()

            logger.log(epoch=epoch+1, step=i+1, d_loss=d_loss.item(), g_loss=g_loss.item(), D_X=real_score.mean().item(), D_G_Z=fake_score.mean().item())

            if (i+1) % 30 == 0:
                sample_log(images, fake_images)

        sample_log(images, fake_images)

        torch.save(G, os.path.join(output_dir, 'modelG-{}'.format(epoch + 1)))
        torch.save(D, os.path.join(output_dir, 'modelD-{}'.format(epoch + 1)))
