#Joseph de la Viesca
#Single-tone testing for V2S200D Voice vibration sensor

import os
import numpy as np
import librosa as lr
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

#Set file directories for plots and recordings
plots_dir = "./plots"
os.makedirs(plots_dir, exist_ok=True)  # Create folder if it doesn't exist

recs_dir = "./recordings"
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
duration = 5 #5 second recording
channels = 2 #Stereo recording

#Generate a sinusoid at each frequency in Hz, playing for a 5 sec duration
frequencies = [100, 500, 1000, 2000, 3000, 5000, 7000, 10000]
all_tones = []


for f_i in frequencies:
    tone = lr.tone(f_i, duration = duration, sr = Fs)
    
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
   
    #Split channels into separate arrays 
    rec_bone = recording[:,0]
    rec_air = recording[:,1]

    #Write recorded data to separate files
    write(filename = os.path.join(recs_dir, f"{f_i}Hz tone (bone conduction mic recording).wav"), rate = Fs, data = rec_bone)
    write(filename = os.path.join(recs_dir, f"{f_i}Hz tone (air mic recording).wav"), rate = Fs, data = rec_air)

    #Generate spectrograms
    f_air_plot, t_air, Sxx_air_plot = spectrogram(rec_air, fs=Fs, nperseg=1024)
    f_bone_plot, t_bone, Sxx_bone_plot = spectrogram(rec_bone, fs=Fs, nperseg=1024)
    
    #Plot waveforms for each
    snapshot_size = int(5.0*Fs/f_i)
    time_axis = np.linspace(0, len(rec_bone)/Fs, num = len(rec_bone))
    fig, axes = plt.subplots(2, 2, figsize=(20, 9), gridspec_kw={'height_ratios': [1, 1.2]})

    #Waveforms
    axes[0, 0].plot(time_axis[Fs*3: Fs*3 + snapshot_size], rec_air[Fs*3: Fs*3 + snapshot_size], color='blue')
    axes[0, 0].set_title("Air mic - Waveform")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)

    axes[0, 1].plot(time_axis[Fs*3: Fs*3 + snapshot_size], rec_bone[Fs*3: Fs*3 + snapshot_size], color='green')
    axes[0, 1].set_title("V2S200D Voice Vibration Sensor - Waveform")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].grid(True)

    #Remove zero frequency to avoid log(0)
    nonzero_mask_air = f_air_plot > 0
    f_air_plot = f_air_plot[nonzero_mask_air]
    Sxx_air_plot = Sxx_air_plot[nonzero_mask_air, :]

    nonzero_mask_bone = f_bone_plot > 0
    f_bone_plot = f_bone_plot[nonzero_mask_bone]
    Sxx_bone_plot = Sxx_bone_plot[nonzero_mask_bone, :]

    #Plot Air Mic spectrogram
    pcm1 = axes[1, 0].pcolormesh(t_air, f_air_plot, 10 * np.log10(Sxx_air_plot + 1e-10), shading='gouraud', cmap='viridis')
    axes[1, 0].set_yscale("log")  #Set log scale
    axes[1, 0].set_title("Air Mic - Spectrogram")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Frequency [Hz]")
    fig.colorbar(pcm1, ax=axes[1, 0], format='%+2.0f dB')

    #Plot Bone Mic spectrogram
    pcm2 = axes[1, 1].pcolormesh(t_bone, f_bone_plot, 10 * np.log10(Sxx_bone_plot + 1e-10), shading='gouraud', cmap='viridis')
    axes[1, 1].set_yscale("log")  #Set log scale
    axes[1, 1].set_title("V2S200D Voice Vibration Sensor - Spectrogram")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Frequency [Hz]")
    fig.colorbar(pcm2, ax=axes[1, 1], format='%+2.0f dB')


    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{f_i}Hz tone waveform zoom at 3s + spectrogram.png"))






