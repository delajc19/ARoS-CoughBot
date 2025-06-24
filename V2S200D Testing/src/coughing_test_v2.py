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

#Import cough audio 

cough_filename = "../cough audio/speaking and coughing - iPhone recording.wav"
coughing_audio, sr = lr.load(cough_filename, sr = Fs)
coughing_audiodata = np.array(coughing_audio)

#Recording setup
duration = 4 #8 second recording (duration + duration)
channels = 2 #Stereo recording


print("Beginning test.")

recording = sd.playrec(
    coughing_audio.astype(np.float32), 
    samplerate = Fs, 
    channels = channels, 
    dtype = "float32", 
    device = (input_device, output_device))
sd.wait()

#File naming scheme: Noise level _ Sound
        #Noise levels: Quiet, Low Noise, High Noise
        #Sounds: Coughing + Speaking
write(filename = os.path.join(recs_dir, "HN_C+S.wav"), rate = Fs, data = recording)

print("Test complete!")