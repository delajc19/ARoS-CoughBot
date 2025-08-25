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
# Generate AWGN with specified mean and standard deviation
def generate_awgn(signal_length, mean=0, std_dev=1):
    """
    Generate Additive White Gaussian Noise
    
    Parameters:
    signal_length: length of the noise vector
    mean: mean of the Gaussian distribution (default: 0)
    std_dev: standard deviation of the Gaussian distribution (default: 1)
    
    Returns:
    noise: AWGN vector
    """
    return np.random.normal(mean, std_dev, signal_length)
#Specify desired input and output device names
input_device_name = "USB V2S-Demo"
output_device_name = "Speakers/Headphones (Realtek(R)"

input_device = find_device(input_device_name, kind='input')
output_device = find_device(output_device_name, kind='output')
print("Using input device: " + str(sd.query_devices()[input_device]['name']))
print("Using output device: "+ str(sd.query_devices()[output_device]['name']))

Fs = 48000 #set sample rate

numtrials = 20

len_t = 10
len_n = int(Fs*len_t)

recordings = np.array(np.zeros(shape = (numtrials, len_n,2)), dtype = np.float32)
noise = generate_awgn(len_n, mean = 0, std_dev = 1)


#Play tones and record simultaneously
for i in range(numtrials):
    print(f"Trial #{i+1}")
    rec = sd.playrec(
        data = noise.astype(np.float32),
        samplerate= Fs,
        channels = 2,
        dtype = 'float32',
        device = (input_device, output_device)
    )
    sd.wait()
    recordings[i] = rec

#write each trial to a .wav file
template = "8-23-25_datacollection_sensitivity_anechoic_noise_trial"

for i in range(numtrials):
    filename = f"{template}{i+1}.wav"
    write(os.path.join(recs_dir,filename), Fs, recordings[i])









