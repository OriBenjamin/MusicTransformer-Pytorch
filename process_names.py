# create a json file with the following format:
#[
#    {
#        "canonical_composer": "Project",
#        "canonical_title": "Project",
#        "split": "train",
#        "year": 2024,
#        "midi_filename": "1.mid",
#        "audio_filename": "1.wav",
#        "duration": 310.5
#    }
#]
# for each midi file in dataset_with_hands/midis

import pretty_midi
import os
import pickle
import json
import random

JSON_FILE = "midi_info.json"

# create a json file
midi_info = []

# get all the midi files
midi_files = os.listdir("dataset_with_hands/midis")
test_num = 0
train_num = len(midi_files)/3
val_num = 2*len(midi_files)/3
for i, midi_file in enumerate(midi_files):
    midi_path = os.path.join("dataset_with_hands/midis", midi_file)
    print(midi_path)
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # Extract information about the tracks and notes

    midi_info.append({
        "canonical_composer": "Project",
        "canonical_title": "Project",
        "split": "test" if i<train_num else "train" if i<val_num else "validation",
        "year": 2024,
        "midi_filename": midi_file,
        "audio_filename": midi_file.replace(".mid", ".wav"),
        "duration": (midi_data.get_end_time()),
    })

# write the json file
with open(JSON_FILE, "w") as f:
    json.dump(midi_info, f, indent=4)

print("Done")