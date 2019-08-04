import os
from modules import midi
from glob import glob
from multiprocessing import Pool

OUT_DIR = '../data/ginko_npy'
MID_PATTERN = '../data/Ginko_s_Midi_Storage/**/*.mid'

IGNORE_DURATION = False

def use_multiprocess():
    os.makedirs(OUT_DIR, exist_ok=False)
    files = sorted(glob(MID_PATTERN, recursive=True))

    pool = Pool(8)
    pool.map_async(save_single, files)
    pool.close()
    pool.join()
    print('Task ended.')


def save_to_npy():
    os.makedirs(OUT_DIR, exist_ok=False)
    files = sorted(glob(MID_PATTERN, recursive=True))

    for index, file in enumerate(files):
        print(f'[{index + 1}/{len(files)}] {file}')
        save_single(file)

def save_single(midi_file):
    try:
        runner = midi.MidiRunner(midi_file)
        runner.save_to_numpy(midi_file, OUT_DIR, ignore_time=IGNORE_DURATION)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # save_to_npy()
    use_multiprocess()
