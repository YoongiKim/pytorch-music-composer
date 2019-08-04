#%%
from modules import midi

#%%
file = 'vgmusic/Nintendo 08 DS/Blaze.mid'
#%%
runner = midi.MidiRunner(file)
samples = runner.midi_to_samples()
#%%
