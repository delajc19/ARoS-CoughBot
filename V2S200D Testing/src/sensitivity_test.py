import numpy as np
import sounddevice as sd
import librosa as lr
import os
from scipy.io.wavfile import write

recs_dir = "H:\My Drive\ARoS Lab\stereo recordings"
os.makedirs(recs_dir, exist_ok=True)

#Find input and output devices
def find_device(name_substring, kind='input'):
    name_substring = name_substring.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if name_substring in dev['name'].lower():
            if (kind == 'input' and dev['max_input_channels'] > 0) or \
               (kind == 'output' and dev['max_output_channels'] > 0):
                return idx
    raise ValueError(f"No {kind} device matching '{name_substring}' found.")

#Specify desired input and output device names
input_device_name = "USB V2S-Demo"
output_device_name = "Speakers/Headphones (Realtek(R)"

input_device = find_device(input_device_name, kind='input')
output_device = find_device(output_device_name, kind='output')
print("Using input device: " + str(sd.query_devices()[input_device]['name']))
print("Using output device: "+ str(sd.query_devices()[output_device]['name']))

Fs = 48000 #set sample rate

len_t = 2
len_n = int(Fs*len_t)

#Set of frequencies
freqs = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
tones = np.array([], dtype = np.float32)


#Generate tones
for i in range(len(freqs)):
    tones = np.concatenate((tones, lr.tone(frequency=freqs[i], sr=Fs, length=len_n)))

duration_n = len_n*len(freqs)
numtrials = 20
recordings = np.array(np.zeros(shape = (numtrials, duration_n,2)), dtype = np.float32)


#Play tones and record simultaneously
for i in range(numtrials):
    print(f"Trial #{i+1}")
    rec = sd.playrec(
        data = tones.astype(np.float32),
        samplerate= Fs,
        channels = 2,
        dtype = 'float32',
        device = (input_device, output_device)
    )
    sd.wait()
    recordings[i] = rec

#write each trial to a .wav file
template = "8-23-25_datacollection_sensitivity_anechoic_stethospeaker_trial"

for i in range(numtrials):
    filename = f"{template}{i+1}.wav"
    write(os.path.join(recs_dir,filename), Fs, recordings[i])









