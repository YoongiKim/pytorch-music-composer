import pygame
import numpy as np
from modules import midi
import matplotlib.pyplot as plt
import scipy.stats as stats
import cv2
import torch
from modules import model as M
from glob import glob
import random
from tools import numpy_to_midi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "params.ckpt"
BASELINE_MIDI = "baseline.mid"

# Model specific
NUM_SAMPLES = 8
NOTE_H = 96
NOTE_W = 96
H_DIM = 2048
Z_DIM = 128


class Main:
    def __init__(self, model_path=MODEL_PATH, baseline_midi=BASELINE_MIDI):
        self.model_path = model_path
        self.baseline_midi = baseline_midi

        self.model = self.load_model(self.model_path)
        self.on_midi_file_load(self.baseline_midi)
        self.save_music(self.generated)
        self.display_()


    def on_midi_file_load(self, midi_file):
        samples = self.load_samples(midi_file)

        X = torch.from_numpy(samples).unsqueeze(0).float().to(device)
        self.x_reconst, self.mu, self.log_var = self.model_reconstruct(X)

        self.generated, Z = self.generate_music(z_params=np.random.uniform(0.0,0.1,Z_DIM))

        self.org_note_img = self.samples_to_img(X)
        self.dist_img = self.distributions_to_img(self.mu, self.log_var, z_tensor=Z)
        self.gen_note_img = self.samples_to_img(self.generated)

    def generate_music(self, z_params: np.array):
        assert z_params.shape == (Z_DIM, ), "latent vector size mismatch."

        params = torch.from_numpy(z_params).to(device).float()
        std = torch.exp(self.log_var / 2)
        eps = torch.randn_like(std)
        Z = self.mu + eps * std + params
        Z = Z.to(device)
        new_samples = self.model_decode(Z)

        return new_samples, Z

    def display_(self):
        cv2.imshow('original_notes', cv2.cvtColor(self.org_note_img, cv2.COLOR_RGB2BGR))
        cv2.imshow('generated_notes', cv2.cvtColor(self.gen_note_img, cv2.COLOR_RGB2BGR))
        cv2.imshow('distributions', cv2.cvtColor(self.dist_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    @staticmethod
    def samples_to_img(x):
        x = x.clone().detach().cpu().numpy()
        x = x > 0.5
        x = (np.array(x)*255.0).astype(np.uint8)
        x = x.reshape((NUM_SAMPLES * NOTE_H, NOTE_W, 1))
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        x = np.array(x).transpose([1,0,2])
        x = np.flipud(x)
        return x

    @staticmethod
    def distributions_to_img(mu_tensor, log_var_tensor, z_tensor=None):
        mu = mu_tensor.clone().detach().squeeze().cpu().numpy()
        log_var = log_var_tensor.clone().detach().squeeze().cpu().numpy()
        z = z_tensor.clone().detach().squeeze().cpu().numpy()

        fig = plt.figure()
        fig.tight_layout(pad=0)

        sigma = np.sqrt(np.exp(log_var))
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.scatter(z, np.ones_like(z)*8, s=1, marker='.', c='red')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        return data

    @staticmethod
    def load_model(model_path):
        model = M.VAE(NUM_SAMPLES, NOTE_H, NOTE_W, h_dim=H_DIM, z_dim=Z_DIM)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        print(f"Loaded model {model_path}")
        return model

    @staticmethod
    def load_samples(baseline_midi):
        midi_runner = midi.MidiRunner(baseline_midi)
        samples = midi_runner.midi_to_samples()

        n, h, w = samples.shape
        assert n >= NUM_SAMPLES, f"Duration time is too short (n < {NUM_SAMPLES}) - {BASELINE_MIDI}. Please select other file."

        sample_start = random.randint(0, n - NUM_SAMPLES)
        return samples[sample_start:sample_start + NUM_SAMPLES]  # [16, 96, 96]

    def model_reconstruct(self, X):
        with torch.no_grad():
            x_reconst, mu, log_var = self.model(X)
        return x_reconst, mu.squeeze(), log_var.squeeze()

    def model_decode(self, Z):
        Z = Z.unsqueeze(0)
        with torch.no_grad():
            out = self.model.decode(Z)
        return out

    def model_reparameterize(self, mu, log_var):
        mu, log_var = mu.unsqueeze(0), log_var.unsqueeze(0)
        with torch.no_grad():
            z = self.model.reparameterize(mu, log_var)
        return z

    def save_music(self, X, file='generated.mid'):
        x = X.clone().detach().squeeze().cpu().numpy()
        midi.MidiRunner.samples_to_midi(x, file, ticks_per_beat=48)
        print("Saved generated midi - ", file)


if __name__ == '__main__':
    main = Main()
