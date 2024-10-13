import numpy as np
import pandas as pd


# load h5 file
file_path = "dataset_with_hands/h5s/recordings_24_playing_enrollment03_free_playing05.h5"
df = pd.read_hdf(path_or_buf=file_path, key='df')
# midi_file_path = "dataset_with_hands/midis/recordings_01_playing_enrollment02_free_playing01.mid"
# import pretty_midi
# mid = pretty_midi.PrettyMIDI(midi_file=midi_file_path)
# print(mid.instruments[0].notes)
# show some stats
# print(f'Number of samples in file: {len(df)}')
# df_duration = df.iloc[-1].datetime - df.iloc[0].datetime
# print(f'Duration of recording in seconds: {df_duration.total_seconds():.2f}.')

# # device (piano) position during the recording
# print('piano position:')
# print(df['dev_pos'].mean())

# # device (piano) position during the recording
# print('piano position:')
# print(df['dev_pos'].mean())

# # velocities of midi events
# print('velocities of midi events:')
# print(df['velocity'].apply(lambda x: x if x > 0 else np.nan).dropna().values)
df.to_excel('outputProblem.xlsx')
# print(df[['datetime', 'notes', 'velocity']])
# print(mid.instruments[0].notes)
# check
