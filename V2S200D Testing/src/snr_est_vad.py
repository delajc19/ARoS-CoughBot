#Joseph de la Viesca

import os
import numpy as np
from scipy.io.wavfile import read, write
import scipy.signal as signal
import matplotlib.pyplot as plt
from vad import EnergyVAD

#Initialize constants
MAX_INT16 = 32767.0
MAX_INT32 = 2147483647.0

def normalize_audio(audio, dtype):
    if(dtype=="float32"):
        return np.array(audio).astype(np.float32)
    elif(dtype=="int16"):
        return audio/MAX_INT16 #Normalize by max int16
    elif(dtype=="int32"):
        return audio/MAX_INT32 #Normalize by max int32

#Select audio file
audio_dir = "..\\stereo recordings\\"

if not os.path.exists(audio_dir):
    raise FileNotFoundError(f"Directory does not exist: {os.path.abspath(audio_dir)}")

#List files
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


# audiofile = audiofile[:5*Fs]
audio_dtype = audiofile.dtype
tlen = len(audiofile)/Fs
print(f"Fs = {Fs} Hz, Duration = {tlen:.2f} s")
print(f"Type = {audiofile.dtype}")

# #example tone for debugging
# sin_freq = 2000
# tone = lr.tone(frequency = sin_freq, sr = Fs, duration = tlen)

#Split sterero recording into separate channels and normalize to range [-1.0, 1.0]
bone_rec = normalize_audio(audiofile[:,0], audio_dtype) #V2S sensor audio on left channel
air_rec = normalize_audio(audiofile[:,1], audio_dtype) #Air mic audio on right channel

frame_length = 40
frame_shift = frame_length // 4
energy_threshold = 0.01
pre_emphasis = 0.95

vad = EnergyVAD(    sample_rate = Fs,
                    frame_length = frame_length,
                    frame_shift = frame_shift,
                    energy_threshold = energy_threshold,
                    pre_emphasis = pre_emphasis)

voice_activity = vad(bone_rec)
mask = voice_activity

output = np.zeros((len(bone_rec),))
frame_length = Fs*frame_length //1000
frame_shift = Fs*frame_shift //1000
for i in range(voice_activity.shape[0]):
    output[i*frame_shift : i*frame_shift + frame_length] = \
        voice_activity[i]*air_rec[i*frame_shift : i*frame_shift + frame_length]



fig, axes = plt.subplots(3,1)

plt.figure(1)
t = np.linspace(0,tlen,int(48000*tlen))
axes[0].plot(t,air_rec)
axes[0].plot(t, air_rec, alpha=0.5)
axes[0].fill_between(t, -1, 1, where=(output != 0), color='orange', alpha=0.2, label="VAD Active")
axes[0].legend()
n = np.linspace(0,len(voice_activity),len(voice_activity))
axes[1].plot(n,mask)
axes[2].plot(t,output)
plt.show()

audiofile[:,1] = output
audio_dir = "../stereo recordings"
outfilename = f"VAD_{file_list[selection]}"
write(os.path.join(audio_dir, outfilename), Fs, audiofile)

