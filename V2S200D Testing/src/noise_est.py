import os
from scipy import signal
from scipy.io.wavfile import read
import librosa as lr
import numpy as np
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

#Load files
audio_dir = "..\\recordings\\"
filename = "1000Hz tone (bone conduction mic recording).wav" 
file_path = os.path.join(audio_dir, filename)

Fs, bone_rec = read(file_path)
duration = len(bone_rec)/Fs

normalize_audio(bone_rec, bone_rec.dtype)

#Estimate PSD with Welch's method
f, PSD_welch =  signal.welch(bone_rec, Fs, nperseg = 2048)
PSD_welch_dB = 20*np.log10(PSD_welch)
#Generate tone
freq = 1000
tone = lr.tone(freq, duration = duration, sr = Fs)
f2, PSD_clean = signal.welch(tone, Fs, nperseg = 2048)
PSD_clean_dB = 20*np.log10(PSD_clean)

#Plots
t = np.linspace(0, duration, int(duration*Fs))

plt.plot(f, PSD_welch_dB)
plt.plot(f2, PSD_clean_dB)



plt.show()







