import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utilities.constants import *
from utilities.device import cpu_device

files_in_output_dir = os.listdir("output/test/")
i_stream    = open("deans_out/test/output.mid.pickle", "rb")#+files_in_output_dir[0], "rb")
# return pickle.load(i_stream), None
# set encoding to 'latin1' to avoid encoding errors
# raw_mid     = pickle.load(i_stream)
raw_mid     = torch.tensor(pickle.load(i_stream), dtype=TORCH_LABEL_TYPE, device=cpu_device())
i_stream.close()

print(raw_mid[0:20])
print(raw_mid.shape)
# load midi with preety_midi
import pretty_midi
# files_in_output_dir = os.listdir("maestro-v2.0.0/test/2006/")
# midi_data = pretty_midi.PrettyMIDI("maestro-v2.0.0/2006/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_02_Track02_wav.midi")
midi_data = pretty_midi.PrettyMIDI("deans_in/output.mid")
# print(dir(midi_data.instruments[0]))
print(midi_data)
print(midi_data.instruments[0].notes)
# print("notes: ", midi_data.instruments[0].notes[0:20])
# print("control_changes: ", midi_data.instruments[0].control_changes[0:10])
# print("end time: ", midi_data.instruments[0].get_end_time())
# print("shape: ", raw_mid[0].shape)
# print("attributes: ", dir(raw_mid[0]))

# print(files_in_output_dir[0])




# import pretty_midi

# # Load the provided MIDI file to inspect its content
# midi_path = 'deans_in/output.mid'
# midi_data = pretty_midi.PrettyMIDI(midi_path)

# # Extract information about the tracks and notes
# track_info = []
# for instrument in midi_data.instruments:
#     notes_info = [(note.start, note.end, note.pitch, note.velocity) for note in instrument.notes]
#     track_info.append({
#         'instrument_name': instrument.name,
#         'program': instrument.program,
#         'is_drum': instrument.is_drum,
#         'notes': notes_info
#     })

# # Print track information
# for idx, track in enumerate(track_info):
#     print(f"Track {idx + 1}:")
#     print(f"  Instrument Name: {track['instrument_name']}")
#     print(f"  Program: {track['program']}")
#     print(f"  Is Drum: {track['is_drum']}")
#     print(f"  Notes Count: {len(track['notes'])}")
#     for note in track['notes']:
#         print(f"    Note Start: {note[0]}, End: {note[1]}, Pitch: {note[2]}, Velocity: {note[3]}")
