import torch
from torchvision.utils import save_image
from modules import model as M
from modules.dataset import MidiDataset
import os
import nonechucks as nc
from utils.torch_logger import TorchLogger
import numpy as np
from glob import glob
from tensorboardX import SummaryWriter

DATASET_PATH = "data/ginko_npy/**/*.npy"

USE_ADVERSARIAL = False

LOAD_G = True
LOAD_D = False
G_MODEL_PATH = "output/**/G-*.ckpt"
D_MODEL_PATH = "output/**/D-*.ckpt"

height = 96
width = 96
num_samples = 8
num_epochs = 1000
batch_size = 512
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

    g_model_path = sorted(glob(G_MODEL_PATH, recursive=True))
    if len(g_model_path) == 0:
        LOAD_G = False
        start_epoch=0
    else:
        g_model_path = g_model_path[-1]
        start_epoch = int(g_model_path.replace('\\','/').split('/')[-1].split('.')[0].split('-')[1])

    if USE_ADVERSARIAL:
        d_model_path = sorted(glob(D_MODEL_PATH, recursive=True))
        if len(d_model_path) == 0:
            LOAD_D = False
        else:
            d_model_path = d_model_path[-1]


    os.makedirs(output_dir, exist_ok=True)

    dataset = nc.SafeDataset(MidiDataset(npy_glob_pattern=DATASET_PATH, num_samples=num_samples, train_step_multiplier=100))
    dataloader = nc.SafeDataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    writer = SummaryWriter('logs')

    G = M.VAE(num_samples, height, width, h_dim, z_dim)

    if USE_ADVERSARIAL:
        D = M.Discriminator(num_samples, height, width)

    if LOAD_G:
        G.load_state_dict(torch.load(g_model_path))
        print(f"Loaded model {g_model_path}, epoch={start_epoch}")
    else:
        print("Training generator(VAE) from scratch")

    if LOAD_D and USE_ADVERSARIAL:
        D.load_state_dict(torch.load(d_model_path))
        print("Loaded discriminator ", d_model_path)
    elif not LOAD_D and USE_ADVERSARIAL:
        print("Training discriminator from scratch")

    G = G.to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-3)
    if USE_ADVERSARIAL:
        D = D.to(device)
        d_opt = torch.optim.Adam(D.parameters(), lr=1e-3)

    criterion = torch.nn.BCELoss(reduction='sum')

    total_step = len(dataset) // batch_size
    logger = TorchLogger(num_epochs, total_step, summary_writer=writer)

    for epoch in range(start_epoch, num_epochs):
        logger.reset()

        for i, x in enumerate(dataloader):
            x = x.to(device)

            real_labels = torch.ones(x.size(0), 1).to(device)
            fake_labels = torch.zeros(x.size(0), 1).to(device)

            if USE_ADVERSARIAL and (i < 10 or d_loss.item() > g_loss.item()):
                # Create the labels which are later used as input for the BCE loss

                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #

                # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # Second term of the loss is always zero since real_labels == 1
                outputs = D(x)
                d_loss_real = criterion(outputs, real_labels) / batch_size
                real_score = outputs

                # Compute BCELoss using fake images
                # First term of the loss is always zero since fake_labels == 0
                # z = torch.randn_like(images).to(device)

                x_reconst, mu, log_var = G(x)
                outputs = D(x_reconst.detach())
                d_loss_fake = criterion(outputs, fake_labels) / batch_size
                fake_score = outputs

                # Backprop and optimize
                d_loss = (d_loss_real + d_loss_fake) * 1000
                d_opt.zero_grad()
                d_loss.backward()
                # torch.nn.utils.clip_grad_norm_(D.parameters(), 0.001)
                d_opt.step()
            else:
                x_reconst, mu, log_var = G(x)

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            reconst_loss = criterion(x_reconst, x) / batch_size
            quantized_reconst_loss = criterion((x_reconst>0.5).float(), x) / batch_size / 10
            kl_div = (-torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / batch_size

            loss = reconst_loss + quantized_reconst_loss + kl_div

            if USE_ADVERSARIAL:
                outputs = D(x_reconst.detach())
                g_loss = criterion(outputs, real_labels) / batch_size * 2000
                loss += g_loss

            g_opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(G.parameters(), 0.001)
            g_opt.step()

            if USE_ADVERSARIAL:
                logger.log(epoch=epoch+1, step=i+1, reconst_loss=reconst_loss.item(), q_reconst_loss=quantized_reconst_loss.item(), kl_div=kl_div.item(),
                        d_loss=d_loss.item(), g_loss=g_loss.item(), total_loss=loss.item(), D_X=real_score.mean().item(), D_G_Z=fake_score.mean().item())
            else:
                logger.log(epoch=epoch + 1, step=i + 1, reconst_loss=reconst_loss.item(), q_reconst_loss=quantized_reconst_loss.item(), kl_div=kl_div.item())

        logger.update_tensorboard(epoch + 1)

        if epoch % 1 == 0:
            with torch.no_grad():
                z = torch.randn(batch_size, z_dim).to(device)
                out = G.decode(z)
                save_sample_images(out, epoch + 1, i+1, 'sample', filter=True)
                save_sample_music(out, epoch+1, i+1, 'sample')
                save_sample_music(x, epoch+1, i+1, name='real')

                x_reconst, mu, log_var = G(x)
                save_sample_images(x, epoch + 1, i+1, 'real', filter=True)
                save_sample_images(x_reconst, epoch + 1, i + 1, 'reconst', filter=True)
                save_sample_music(x_reconst, epoch + 1, i + 1, 'reconst')

        if epoch % 5 == 0:
            torch.save(G.state_dict(), os.path.join(output_dir, 'G-{}.ckpt'.format(epoch + 1)))
            if USE_ADVERSARIAL:
                torch.save(D.state_dict(), os.path.join(output_dir, 'D-{}.ckpt'.format(epoch + 1)))

    torch.save(G.state_dict(), os.path.join(output_dir, 'G-{}.ckpt'.format(epoch + 1)))
    if USE_ADVERSARIAL:
        torch.save(D.state_dict(), os.path.join(output_dir, 'D-{}.ckpt'.format(epoch + 1)))
