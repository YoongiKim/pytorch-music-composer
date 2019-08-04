import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from modules import model
from modules.dataset import MidiDataset
import os
import nonechucks as nc
from utils.torch_logger import TorchLogger
import numpy as np
from glob import glob

LOAD_MODEL = True
# MODEL_PATH = "output/**/params*"
MODEL_PATH = "params*.ckpt"

DATASET_PATH = "data/ginko_npy/**/*.npy"

height = 96
width = 96
num_samples = 8
num_epochs = 1000
batch_size = 128
output_dir = 'output'
h_dim=2048
z_dim=128


def save_sample_images(images, epoch, step, name='real_images', filter=True):
    x = images
    if filter:
        x = x > 0.5
    x = x.reshape(-1, 1, num_samples * height, width)
    save_image(x[0:1], os.path.join(output_dir, '{}-{}-{}.png'.format(name, epoch, step)))

def save_sample_music(images, epoch, step, name='sample'):
    x = images
    from tools import numpy_to_midi
    x = x.reshape(-1, num_samples, height, width)
    npy_file = os.path.join(output_dir, '{}-{}-{}.npy'.format(name, epoch, step))
    np.save(npy_file, x.detach().cpu().numpy()[0])
    numpy_to_midi.numpy_to_midi(npy_file, npy_file + '.mid', ticks_per_beat=48)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = sorted(glob(MODEL_PATH, recursive=True))
    if len(model_path) == 0:
        LOAD_MODEL = False
    else:
        model_path = model_path[-1]


    os.makedirs(output_dir, exist_ok=True)

    dataset = nc.SafeDataset(MidiDataset(npy_glob_pattern=DATASET_PATH, num_samples=num_samples, train_step_multiplier=100))
    dataloader = nc.SafeDataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = model.VAE(num_samples, height, width, h_dim, z_dim)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(model_path))
        print("Loaded model ", model_path)
    else:
        print("Training from scratch")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_step = len(dataset) // batch_size
    logger = TorchLogger(num_epochs, total_step)

    for epoch in range(num_epochs):
        logger.reset()

        for i, x in enumerate(dataloader):
            x = x.to(device)

            x_reconst, mu, log_var = model(x)

            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False) / batch_size
            quantized_reconst_loss = F.binary_cross_entropy((x_reconst>0.5).float(), x, size_average=False) / batch_size
            kl_div = (-torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / batch_size

            loss = reconst_loss + quantized_reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            logger.log(epoch=epoch+1, step=i+1, reconst_loss=reconst_loss.item(), q_reconst_loss=quantized_reconst_loss.item(), kl_div=kl_div.item())

        if epoch % 1 == 0:
            with torch.no_grad():
                z = torch.randn(batch_size, z_dim).to(device)
                out = model.decode(z)
                save_sample_images(out, epoch + 1, i+1, 'sample', filter=True)
                save_sample_music(out, epoch+1, i+1, 'sample')

                x_reconst, mu, log_var = model(x)
                save_sample_images(x, epoch + 1, i+1, 'real', filter=True)
                save_sample_images(x_reconst, epoch + 1, i + 1, 'reconst', filter=True)
                save_sample_music(x_reconst, epoch + 1, i + 1, 'reconst')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, 'params-{}.ckpt'.format(epoch + 1)))

    torch.save(model.state_dict(), os.path.join(output_dir, 'params-{}.ckpt'.format(epoch + 1)))
