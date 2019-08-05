import numpy as np
from modules import midi
from glob import glob


def numpy_to_midi(npy_file, out_midi_file, ticks_per_beat=48):
    samples = np.load(npy_file)

    midi.MidiRunner.samples_to_midi(samples, out_midi_file, ticks_per_beat)


if __name__ == '__main__':
    files = glob('../output/**/*.npy', recursive=True)
    for file in files:
        print(file)
        numpy_to_midi(file, file+'.mid', ticks_per_beat=48)
