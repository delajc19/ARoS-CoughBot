import os
from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq
from scipy import signal
import librosa as lr
from matplotlib.widgets import Button
import numpy as np
import matplotlib
import sounddevice as sd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

#%%Computations

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

def select_audio_files(n, title="Select a Stereo Audio File"):
    """
    Open a file dialog to select `n` audio files in order.
    
    Returns:
        List of file paths in the selected order.
    """
    root = tk.Tk()
    root.withdraw()  # hide the main window
    file_path = filedialog.askopenfilenames(
        title=title,
        filetypes=[("WAV files", "*.wav")],
        multiple=True
    )
    root.destroy()

    if len(file_path) != n:
        raise ValueError(f"Expected {n} files, but got {len(file_path)}")
    return file_path

def sensor_spectral_noise(n_window, window, hop_length, gain):
    Fs, stereonoise = read("H:\My Drive\ARoS Lab\stereo recordings\silence.wav")
    n_air = gain*normalize_audio(stereonoise[:,1],dtype = stereonoise.dtype)
    n_bone = gain*normalize_audio(stereonoise[:,0],dtype = stereonoise.dtype)

    _, N_air_PSD = signal.welch(n_air, Fs, nperseg = n_window, window = window, noverlap=n_window//4)
    _, N_bone_PSD = signal.welch(n_bone, Fs, nperseg = n_window, window = window, noverlap=n_window//4)

    N_air_amp = np.average(abs(lr.stft(n_air, window = window, n_fft = n_window, hop_length = hop_length)))
    N_bone_amp = np.average(abs(lr.stft(n_bone, window = window, n_fft = n_window, hop_length = hop_length)))

    return N_air_amp, N_air_PSD, N_bone_amp, N_bone_PSD

def align(air_rec, bone_rec):
    #zero mean
    bone_rec = bone_rec - np.mean(bone_rec)
    air_rec = air_rec - np.mean(air_rec)

    Rxx_air = signal.correlate(air_rec, air_rec, mode = 'full')
    Rxy_bone = signal.correlate(bone_rec, air_rec, mode = 'full')
    delay = np.argmax(Rxy_bone) - np.argmax(Rxx_air)
    print(f"Delay estmation: {delay*1000.0/Fs}")

    if(delay > 0):
        bone_rec = bone_rec[delay:]
        bone_rec = np.pad(bone_rec, (0, delay))
    elif(delay < 0):
        delay = -1*delay
        air_rec = air_rec[delay:]
        air_rec = np.pad(air_rec, (0, delay))
    return air_rec, bone_rec

def butter_bp(lowcut, highcut, fs, order):
    w1 = 2*lowcut / fs
    w2 = 2*highcut/ fs
    b, a = signal.butter(order,[w1, w2], btype = 'bandpass', analog = False)
    return b, a
    

#Select audio file

filename = select_audio_files(n = 1)[0]

#Load selected audio file
Fs, audiofile = read(filename)
print(f"Currently viewing: {filename}")

#Ensure user selects stereo file
while(audiofile.ndim != 2 or audiofile.shape[1] < 2):
    raise ValueError("Expected a stereo audio file with 2 channels")


# audiofile = audiofile[:Fs]
audio_dtype = audiofile.dtype
tlen = len(audiofile)/Fs
print(f"Fs = {Fs} Hz, Duration = {tlen:.2f} s")
print(f"Type = {audiofile.dtype}")

#Split sterero recording into separate channels and normalize to range [-1.0, 1.0]
bone_rec = normalize_audio(audiofile[:,0], audio_dtype) #V2S sensor audio on left channel
air_rec = normalize_audio(audiofile[:,1], audio_dtype) #Air mic audio on right channel

#align bone recording and air recording 
# air_rec, bone_rec = align(air_rec,bone_rec)


#Normalized gain
peak_air = max(abs(air_rec))
peak_bone = max(abs(bone_rec))
gain = peak_air/peak_bone
bone_rec = gain*bone_rec

 #%% Filtering

# #filter out 7kHz resonance from the V2S200D mic
# f0 = 7000 #set Q factor and resonant frequency of notch filter
# Q_factor = 1
# b, a = signal.iirnotch(w0 = f0, Q = Q_factor, fs = Fs)
# alpha = 1 #filter influence
# bone_rec = alpha*signal.filtfilt(b, a, x = bone_rec) + (1-alpha)*bone_rec

# #bandpass filter to focus analysis window
# b, a = butter_bp(60,21000,Fs,5)
# air_rec = signal.filtfilt(b, a, air_rec)
# bone_rec= signal.filtfilt(b, a, bone_rec)


#Delay estimation
#Compute cross correlation of air and bone recording
#Use argmax to find delay

#%%Welch's Methon PSD calculation

#Estimate PSD using Welch's method
window = 'hann' #window type
n_window = 16384    #window size
hop_length = n_window // 4

f_air, Pxx_air = signal.welch(air_rec, Fs, nperseg = n_window, window = window, noverlap=n_window//4)
f_bone, Pxx_bone = signal.welch(bone_rec, Fs, nperseg = n_window, window = window, noverlap=n_window//4)

#%% Amplitude spectrum FFT
N = n_window
x = air_rec[:N] * signal.get_window(window, N)  # apply same window
X = fft(x)
freqs = fftfreq(N, 1/Fs)

pos_mask = freqs >= 0
fftfreqs_air = freqs[pos_mask]
fftamps_air = 2.0/N * np.abs(X[pos_mask])  # scale for amplitude spectrum

x = bone_rec[:N] * signal.get_window(window, N)  # apply same window
X = fft(x)
freqs = fftfreq(N, 1/Fs)

pos_mask = freqs >= 0
fftfreqs_bone = freqs[pos_mask]
fftamps_bone = 2.0/N * np.abs(X[pos_mask])  # scale for amplitude spectrum

# #smooth PSDs
# kernel = np.ones((n_window//2048,))
# Pxx_air = np.convolve(Pxx_air, kernel, mode='same')
# Pxx_bone = np.convolve(Pxx_bone, kernel, mode='same')


#load noise spectra
# N_air_amp, N_air_PSD, N_bone_amp, N_bone_PSD = sensor_spectral_noise(n_window=n_window, window=window, hop_length=hop_length, gain = 1)

#Compute STFT for spectrogram
Sxx_air = lr.stft(air_rec, window = window, n_fft = n_window, hop_length = hop_length)
Sxx_bone = lr.stft(bone_rec, window = window, n_fft = n_window, hop_length = hop_length)

#Subtract noise spectra
# Sxx_air = Sxx_air - N_air_amp
# Sxx_bone = Sxx_bone - N_bone_amp

# Pxx_air_new = Pxx_air - N_air_PSD
# Pxx_bone_new = Pxx_bone - N_bone_PSD

#convert after noise subtraction
# air_rec = lr.istft(Sxx_air, window = window, n_fft = n_window, hop_length = hop_length)
# bone_rec = lr.istft(Sxx_bone, window = window, n_fft = n_window, hop_length = hop_length)

HSxx = Sxx_air/Sxx_bone

#Convert to dB
Sxx_air_dB = lr.amplitude_to_db(abs(Sxx_air), ref = np.max)
Sxx_bone_dB = lr.amplitude_to_db(abs(Sxx_bone), ref = np.max)

# ref_val = np.max([np.max(np.abs(N_air_PSD)), np.max(np.abs(Pxx_air)), np.max(np.abs(Pxx_air_new))])

ref_air = np.max(Pxx_air)
ref_bone = np.max(Pxx_bone)

Pxx_air_dB = 10*np.log10(np.abs(Pxx_air)/ref_air) #Convert to dB
Pxx_bone_dB = 10*np.log10(np.abs(Pxx_bone)/ref_bone)


#%%Plot
fig, axes = plt.subplots(3, 2, figsize=(20, 10), gridspec_kw={'height_ratios': [1, 1, 1.5]}, constrained_layout = True)

#Time axis for waveforms
t_x = np.arange(len(air_rec)) / Fs
# Create button axes relative to the waveform axes

# Get position of waveform axes in figure coordinates
pos_air = axes[0,0].get_position()
pos_bone = axes[0,1].get_position()

# Create smaller axes for buttons just below each waveform subplot
# button_width = pos_air.width * 0.3
# button_height = 0.04

# button_ax_air = fig.add_axes([pos_air.x0 - 0.08, pos_air.y0 + 0.2, button_width, button_height])
# button_ax_bone = fig.add_axes([pos_bone.x0, pos_bone.y0 + 0.2, button_width, button_height])

# Create buttons
# button_air = Button(button_ax_air, '▶ Play Air Mic')
# button_bone = Button(button_ax_bone, '▶ Play Bone Mic')

# Define play callbacks (assuming air_rec, bone_rec, Fs already exist)
# def play_air(event):
#     sd.stop()
#     sd.play(air_rec, Fs)

# def play_bone(event):
#     sd.stop()
#     sd.play(bone_rec, Fs)

# button_air.on_clicked(play_air)
# button_bone.on_clicked(play_bone)

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

#Air Microphone spectrum
# ref_val = np.max([np.max(np.abs(N_air_PSD)), np.max(np.abs(Pxx_air))])

axes[1,0].plot(f_air, Pxx_air_dB)
# axes[1,0].semilogy(fftfreqs_air, np.pow(fftamps_air,2)/fftfreqs_air)
# axes[1,0].plot(f_air,10*np.log10(np.abs(N_air_PSD)/ref_val))
# axes[1,0].plot(f_air,Pxx_air_new_dB)
axes[1,0].set_title("Air Mic Spectrum")
axes[1,0].set_ylabel("Power Spectral Density [dB]")
axes[1,0].set_xlabel("Frequency [Hz]")
axes[1,0].set_xlim([0,10000])
# axes[1,0].set_ylim([-30, 0])
axes[1,0].grid(True)
# axes[1,0].legend(["PSD","Noise PSD", "Speech - Noise PSD"])

#Bone Microphone spectrum

axes[1,1].plot(f_bone, Pxx_bone_dB, color = green)
# axes[1,1].semilogy(fftfreqs_bone, np.pow(fftamps_bone,2)/fftfreqs_bone, color = green)

# axes[1,1].plot(f_bone,10*np.log10(np.abs(N_bone_PSD)/ref_val))
# axes[1,1].plot(f_bone,Pxx_bone_new_dB)
axes[1,1].set_title("Bone Mic Spectrum")
axes[1,1].set_ylabel("Power Spectral Density [dB]")
axes[1,1].set_xlabel("Frequency [Hz]")
axes[1,1].set_xlim([0,10000])
# axes[1,1].set_ylim([-30, 0])
axes[1,1].grid(True)
# axes[1,1].legend(["PSD","Noise PSD", "Speech - Noise PSD"])

#Set up freq and time axes for spectrogram
freqs = lr.fft_frequencies(sr = Fs, n_fft = n_window)
times = lr.frames_to_time(np.arange(Sxx_bone.shape[1]), sr=Fs, hop_length=hop_length, n_fft=n_window)

#Air microphone spectrogram
cmap = 'inferno'
# Air microphone spectrogram (log freq axis)
pcm1 = lr.display.specshow(Sxx_air_dB,
                                sr=Fs,
                                hop_length = hop_length,
                                n_fft = n_window,
                                x_axis='time',
                                y_axis='log',
                                cmap=cmap,
                                ax=axes[2,0])
axes[2,0].set_title("Air Mic Spectrogram [Hz]")
axes[2,0].set_xlabel("Time [s]")
axes[2,0].set_ylabel("Frequency [Hz]")

# Bone microphone spectrogram (log freq axis)
pcm2 = lr.display.specshow(Sxx_bone_dB,
                                sr=Fs,
                                hop_length = hop_length,
                                n_fft = n_window,
                                x_axis='time',
                                y_axis='log',
                                cmap=cmap,
                                ax=axes[2,1])
axes[2,1].set_title("V2S200D Bone Conduction Mic Spectrogram [Hz]")
axes[2,1].set_xlabel("Time [s]")
axes[2,1].set_ylabel("Frequency [Hz]")


#Add colorboar for spectrogram plots
fig.colorbar(pcm1, ax=axes[2,0], format='%+2.0f dB')
fig.colorbar(pcm2, ax=axes[2,1], format='%+2.0f dB')
fig.subplots_adjust(hspace = 0.3)

# #specify custom tick marks for log y axis
# f_ticks = [100, 500, 1000, 2000, 3000, 5000, 10000, 20000]
# axes[2,0].set_yticks(f_ticks)
# axes[2,0].set_yticklabels([f"{f}" for f in f_ticks])

# axes[2,1].set_yticks(f_ticks)
# axes[2,1].set_yticklabels([f"{f}" for f in f_ticks])

#Add supertitle to display file name
fig.suptitle(f"Currently viewing: {filename}")


plt.show()



# %%
