from torch.utils import data
from glob import glob
import numpy as np
import random


class MidiDataset(data.Dataset):
    def __init__(self, npy_glob_pattern='vgmusic_npy/**/*.npy', num_samples=16):
        self.num_samples = num_samples
        self.files = sorted(glob(npy_glob_pattern, recursive=True))

    def __getitem__(self, index):
        try:
            samples = np.load(self.files[index]).astype(np.float32)
        except Exception as e:
            raise IOError(f"Data corrupted: {self.files[index]} - {e}")

        n, h, w = samples.shape

        assert n >= self.num_samples, f"Duration too short (n < {self.num_samples}) - {self.files[index]}"

        sample_start = random.randint(0, n - self.num_samples)
        return samples[sample_start:sample_start+self.num_samples]  # [16, 96, 96]

    def __len__(self):
        return len(self.files)

    def filter_file(self, file):
        try:
            samples = np.load(file)
        except Exception as e:
            print(f"Data corrupted: {file} - {e}")
            return False

        n, h, w = samples.shape
        if n < self.num_samples:
            print(f"Ignore: Duration too short (n < {self.num_samples}) - {file}")
            return False

        return True


if __name__ == '__main__':
    dataset = MidiDataset()

    image = dataset.__getitem__(0)
    print(image.shape)
