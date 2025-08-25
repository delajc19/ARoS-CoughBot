import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import os

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

#-----MAIN SCRIPT-----

#Read in files
Fs = 48000                       #sample rate in Hz
numtones = 14                    #number of tones played
len_t = 3                        #length of each tone in sec
duration_n = numtones*len_t*Fs   #duration of recording in samples
numtrials = 20                   #number of trials

recordings = np.array(np.zeros((numtrials, duration_n), dtype = np.float32))

recs_dir = "H:\\My Drive\\ARoS Lab\\stereo recordings"
template = "8-22-25_datacollection_sensitivity_stethospeaker_trial"


for i in range(numtrials):
    filename = os.path.join(recs_dir, f"{template}{i+1}.wav")
    Fs, curr_rec = read(filename)
    recordings[i,:] = normalize_audio(curr_rec[:,0], dtype = curr_rec.dtype)

#Estimate PSD using Welch's method
window = 'hann' #window type
n_window = 16384    #window size
hop_length = n_window // 4

f, Pxx = signal.welch(recordings[0], Fs, nperseg = n_window, window = window, noverlap=n_window//4)

spectra = np.array(np.zeros((numtrials, len(f)), dtype = 'float32'))
for i in range(numtrials):
    _, spectra[i,:] = signal.welch(recordings[i], Fs, nperseg = n_window, window = window, noverlap=n_window//4)

#Reference 0dB at 1kHz
target_freq = 1000  # Hz
freqs = [0, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
f_idx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(len(freqs)):
    f_idx[i] = np.argmin(np.abs(f - freqs[i]))


#Average all recordings
sampled_avg = np.array(np.zeros((numtrials, len(freqs)), dtype = 'float32'))
avg_spectrum = np.array(np.zeros((len(Pxx)), dtype = 'float32'))

# Calculate the mean spectrum across all trials
avg_spectrum = np.mean(spectra, axis=0)

for i in range(len(freqs)):
    sampled_avg[i] = avg_spectrum[f_idx[i]]


        
f = interp1d(freqs, sampled_avg, kind = 'cubic')


fnew = np.linspace(0,10000,200)

idx_1000 = np.argmin(np.abs(f - target_freq))
ref_val = avg_spectrum[idx_1000]

avg_dB = 10*np.log10(np.abs(sampled_avg)/ref_val)
interp_avg_dB = 10*np.log10(np.abs(f(fnew)/ref_val))
plt.plot(freqs,avg_dB,'o')
plt.plot(fnew, f(fnew))

plt.show()



