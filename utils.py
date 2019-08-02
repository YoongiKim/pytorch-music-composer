import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

import midi
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def visualize_samples(samples: np.array):
        print(samples.shape)
        vis_samples = np.concatenate(samples, axis=0)
        vis_samples = vis_samples.transpose([1, 0])
        print(vis_samples.shape)

        cv2.imshow("samples", vis_samples*255)
        cv2.waitKey(0)

if __name__ == '__main__':
    file = 'vgmusic/Nintendo 08 DS/DESTINY.mid'

    runner = midi.MidiRunner(file)
    samples = runner.midi_to_samples()

    vis = Visualizer()
    vis.visualize_samples(samples)