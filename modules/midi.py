"""
Original source code from https://github.com/HackerPoet/Composer/blob/master/midi.py
Edited by YoongiKim https://github.com/YoongiKim
"""


from mido import MidiFile, MidiTrack, Message
import numpy as np
import math
import warnings
from glob import glob
import os
from shutil import copy2

num_notes = 96
samples_per_measure = 96

filter_channels = [10]

"""
samples data structure

axis0 -> bar (split of music)
axis1 -> time
axis2 -> note
"""

class MidiRunner:
    def __init__(self, file):
        self.mid = MidiFile(file)
        self.ppq, self.bpm, self.millis, self.ticks_per_measure, self.multi_time_sig, self.multi_tempo\
            = self.get_info()
        self.duration = self.mid.length


    def get_info(self):
        ppq = self.mid.ticks_per_beat
        tempo = 50000
        time_sig = 4/4

        has_tempo = False
        has_time_sig = False
        has_multi_tempo = False
        has_multi_time_sig = False

        for track_idx, track in enumerate(self.mid.tracks[0:1]):
            for event_index, msg in enumerate(track):
                if msg.type == 'set_tempo':
                    if has_tempo and msg.tempo != tempo:
                        has_multi_tempo = True
                        warnings.warn("Detected multiple tempo.")
                    tempo = msg.tempo
                    has_tempo = True

                if msg.type == 'time_signature':
                    if has_time_sig and time_sig != (msg.numerator / msg.denominator):
                        has_multi_time_sig = True
                        warnings.warn("Detected multiple time signature.")

                    time_sig = msg.numerator / msg.denominator
                    has_time_sig = True

        bpm = 60 / (tempo / 1e+6)
        millis_per_tick = 60000 / (bpm * ppq)
        ticks_per_measure = 4 * ppq * time_sig

        return ppq, bpm, millis_per_tick, ticks_per_measure, has_multi_time_sig, has_multi_tempo

    def midi_to_samples(self, ignore_time=False):
        # note_on channel=1 note=44 velocity=127 time=816
        # note_off channel=1 note=44 velocity=64 time=24

        all_notes = {}
        for track_idx, track in enumerate(self.mid.tracks):
            if track_idx in filter_channels:
                continue

            abs_time = 0

            for event_idx, msg in enumerate(track):
                abs_time += msg.time
                sample_time = abs_time * samples_per_measure / self.ticks_per_measure

                if msg.type == 'note_on':
                    if msg.velocity == 0:
                        continue

                    note = msg.note - (128 - num_notes) / 2
                    if not 0 <= note < num_notes:
                        warnings.warn("note out of range")
                        continue

                    if note not in all_notes:
                        all_notes[note] = []
                    else:
                        single_note = all_notes[note][-1]
                        if len(single_note) == 1:  # If already note_on then end that note
                            single_note.append(sample_time)

                    all_notes[note].append([sample_time])

                elif msg.type == 'note_off':
                    note = msg.note - (128 - num_notes) / 2
                    if len(all_notes[note][-1]) != 1:
                        continue
                    all_notes[note][-1].append(sample_time)

        for note in all_notes:
            for start_end in all_notes[note]:
                if len(start_end) == 1:  # If note_on and not closed then end that note
                    start_end.append(start_end[0]+1)

        samples = []
        for note in all_notes:
            for start, end in all_notes[note]:
                sample_ix = int(start / samples_per_measure)
                while len(samples) <= sample_ix:
                    samples.append(np.zeros([samples_per_measure, num_notes], dtype=np.uint8))
                sample = samples[sample_ix]
                start_ix = start - sample_ix * samples_per_measure

                if not ignore_time:
                    end_ix = min(end - sample_ix * samples_per_measure, samples_per_measure)
                    while start_ix < end_ix:
                        sample[int(start_ix), int(note)] = 1
                        start_ix += 1
                else:
                    sample[int(start_ix), int(note)]=1

        return np.array(samples)

    def save_to_numpy(self,midi_file_path, out_dir, ignore_time=False):
        samples = self.midi_to_samples(ignore_time=ignore_time)

        parent_dir_name = str(midi_file_path).replace('\\', '/').split('/')[-2]
        midi_name = str(midi_file_path).replace('\\', '/').split('/')[-1]
        output_dir = os.path.join(out_dir, parent_dir_name).replace('\\', '/')
        os.makedirs(output_dir, exist_ok=True)

        file = f'{output_dir}/{midi_name}_tpb{self.mid.ticks_per_beat}.npy'
        np.save(file, samples)

        # midi_new = f'{output_dir}/{midi_name}_converted.midi'
        # self.samples_to_midi(samples, midi_new, ticks_per_beat=self.mid.ticks_per_beat)
        #
        # copy2(midi_file_path, output_dir)

        print(f'Saved {file}')

    @staticmethod
    def samples_to_midi(samples, file, ticks_per_beat=48, thresh=0.5):
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        track.append(Message('program_change', program=4))

        mid.ticks_per_beat = ticks_per_beat
        ticks_per_measure = 4 * ticks_per_beat
        ticks_per_sample = ticks_per_measure / samples_per_measure

        # note_on channel=1 note=44 velocity=127 time=816
        # note_off channel=1 note=44 velocity=64 time=24

        abs_time = 0
        last_time = 0
        for sample in samples:
            for y in range(sample.shape[0]):
                abs_time += ticks_per_sample
                for x in range(sample.shape[1]):
                    note = int(x + (128 - num_notes) / 2)
                    if sample[y, x] >= thresh and (y == 0 or sample[y - 1, x] < thresh):
                        delta_time = abs_time - last_time
                        track.append(Message('note_on', note=note, velocity=int(sample[y,x]*127), time=int(delta_time)))
                        last_time = abs_time
                    elif sample[y, x] < thresh and (y == sample.shape[0] - 1 or sample[y - 1, x] > thresh):
                        delta_time = abs_time - last_time
                        track.append(Message('note_off', note=note, velocity=int(sample[y,x]*127), time=int(delta_time)))
                        last_time = abs_time
        mid.save(file)


if __name__ == '__main__':
    runner = MidiRunner('../data/vgmusic/Nintendo 08 DS/Blaze.mid')
    samples = runner.midi_to_samples(ignore_time=False)
    print(samples.shape)
