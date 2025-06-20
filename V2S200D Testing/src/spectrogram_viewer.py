import os
from scipy.io.wavfile import read
import librosa as lr
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np
import matplotlib

#Remove matplotlib toolbar
matplotlib.rcParams['toolbar'] = 'none'
import matplotlib.pyplot as plt

#Initialize constants
MAX_INT16 = 32767.0
MAX_INT32 = 2147483647.0

#Normalizes 16 or 32 bit int audio data to a float32 dynamic range [-1.0, 1.0]
def normalize_audio(audio, dtype):
    if(dtype=="float32"):
        return np.array(audio).astype(np.float32)
    elif(dtype=="int16"):
        return audio/MAX_INT16 #Normalize by max int16
    elif(dtype=="int32"):
        return audio/MAX_INT32 #Normalize by max int32


#Select audio file
audio_dir = "C:\\Users\\local-admin\\Documents\\VSCode\\NCSU\\ARoS-CoughBot\\V2S200D Testing\\stereo recordings"
file_list = os.listdir(audio_dir)
num_files = len(file_list)

for i in range(num_files):
    print(f"{i+1} {file_list[i]}")


selection = int(input("Please input the index of the audio file you wish to analyze: ")) - 1

#Ensure user inputs valid index
while(selection < 0 or selection >= num_files):
    selection = int(input(f"Invalid input, please input a value in the range [1, {num_files}]: ")) - 1

filename = os.path.join(audio_dir, file_list[selection])

#Load selected audio file
Fs, audiofile = read(filename)

#Ensure user selects stereo file
while(audiofile.ndim != 2 or audiofile.shape[1] < 2):
    raise ValueError("Expected a stereo audio file with 2 channels")

audio_dtype = audiofile.dtype
tlen = len(audiofile)/Fs
print(f"Fs = {Fs} kHz, Duration = {tlen:.2f} s")
print(f"Type = {audiofile.dtype}")

#Split sterero recording into separate channels and normalize to range [-1.0, 1.0]
bone_rec = normalize_audio(audiofile[:,0], audio_dtype) #V2S sensor audio on left channel
air_rec = normalize_audio(audiofile[:,1], audio_dtype) #Air mic audio on right channel

#Compute STFT for spectrogram
Sxx_air = lr.stft(air_rec)
Sxx_bone = lr.stft(bone_rec)

#global_max = max(Sxx_air.max(), Sxx_bone.max())
global_max = 1024
#Convert to dB
Sxx_air_dB = lr.amplitude_to_db(abs(Sxx_air), ref = np.max)
Sxx_bone_dB = lr.amplitude_to_db(abs(Sxx_bone), ref = np.max)


#Plot
fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'height_ratios': [1, 1.5]}, constrained_layout = True)

#Time axis for waveforms
t_x = np.linspace(0, tlen, int(Fs*tlen))


#Air Microphone waveform 
axes[0,0].plot(t_x, air_rec)
axes[0,0].set_title("Air Mic Waveform")
axes[0,0].set_ylabel("D")
axes[0,0].set_ylim([-1.0,1.0])
axes[0,0].grid(True)

#Bone microphone waveform 
green = (29/255, 188/255, 117/255)
axes[0,1].plot(t_x, bone_rec, color = green)
axes[0,1].set_title("V2S200D Bone Conduction Mic Waveform")
axes[0,1].set_ylabel("D")
axes[0,1].set_ylim([-1.0,1.0])
axes[0,1].grid(True)

#Set up freq and time axes for spectrogram
freqs = lr.fft_frequencies(sr = Fs)
times = lr.frames_to_time(np.arange(Sxx_air.shape[1]), sr=Fs)

#Find global maximum for both spectrograms
vmin = min(Sxx_air_dB.min(), Sxx_bone_dB.min())
vmax = max(Sxx_air_dB.max(), Sxx_bone_dB.max())

#Air microphone spectrogram
cmap = 'viridis'

pcm1 = axes[1,0].pcolormesh(times, freqs/1e3, Sxx_air_dB, shading = 'auto', cmap = cmap,)
                           #vmin = vmin, vmax = vmax)
axes[1,0].set_title("Air Mic Spectrogram")
axes[1,0].set_xlabel("Time [s]")
axes[1,0].set_ylabel("Frequency [kHz]")

#Bone microphone waveform
pcm2 = axes[1,1].pcolormesh(times, freqs/1e3, Sxx_bone_dB , shading = 'auto', cmap = cmap,)
                             #vmin = vmin, vmax = vmax)
axes[1,1].set_title("V2S200D Bone Conduction Mic Spectrogram")
axes[1,1].set_xlabel("Time [s]")
axes[1,1].set_ylabel("Frequency [kHz]")

#Add colorboar for spectrogram plots
fig.colorbar(pcm1, ax=axes[1,0], format='%+2.0f dB')
fig.colorbar(pcm2, ax=axes[1,1], format='%+2.0f dB')
fig.subplots_adjust(hspace = 0.3)

#Add supertitle to display file name
fig.suptitle(f"Currently viewing: {file_list[selection]}")

plt.show()


