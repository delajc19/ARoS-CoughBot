#Joseph de la Viesca
#Single-tone testing for V2S200D Voice vibration sensor, without plots or splitting 

import os
import numpy as np
import librosa as lr
import sounddevice as sd
from scipy.io.wavfile import write


recs_dir = "../stereo recordings"
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

#Recording setup
duration = 4 #8 second recording (duration + duration)
channels = 2 #Stereo recording

#Generate a sinusoid at each frequency in Hz, playing for a 5 sec duration
frequencies = [100, 500, 1000, 2000, 3000, 5000, 7000, 10000]
all_tones = []


for f_i in frequencies:
    #Tone plays for 1/2 the duration of the recording
    tone = lr.tone(f_i, duration = duration, sr = Fs)
    tone = np.concatenate((tone, np.zeros(Fs*duration)))
    
    #Play each tone from the speaker and record
    #Syntiant audio streaming box records a stereo .wav with
    #   Left channel: bone conduction microphone
    #   Right channel: air microphone
    recording = sd.playrec(
        tone.astype(np.float32), 
        samplerate = Fs, 
        channels = channels, 
        dtype = "float32", 
        device = (input_device, output_device))
    sd.wait()

    #File naming scheme: Noise level _ Sound
        #Noise levels: Quiet, Low Noise, High Noise
        #Sounds: Single Tone (freq [Hz])
    write(filename = os.path.join(recs_dir, f"HN_ST{f_i}Hz.wav"), rate = Fs, data = recording)