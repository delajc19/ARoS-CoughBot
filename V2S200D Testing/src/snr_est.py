# Joseph de la Viesca
# 
# This script uses a very rudimentary RMS-based approach to estimate and compare the SNRs of 
# air and bone conduction microphones. The input file is selected from the "stereo recordings"
# folder, containing audio data recorded in the "environmental noise tests". The script is 
# designed to process the sinusoidal tone recordings, which are 8 sec recordings comprised of
# 4 seconds of sinusoid + noise (active) and 4 seconds of noise (inactive). 3 seconds of the 
# active and inactive section are extracted, and the RMS for each is calculated. This is then
# used to find an average SNR value for the bone and air conduction microphones at each frequency
# of a desired target sound.  

import os
import numpy as np
from scipy.io.wavfile import read
import scipy.signal as signal

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

#Estimates delay between two audio files and aligns them accordingly
def align(audio1, audio2):
    Rxx_1 = signal.correlate(audio1, audio1, mode = 'full')
    Rxy_2 = signal.correlate(audio2, audio1, mode = 'full')
    delay = np.argmax(Rxy_2) - np.argmax(Rxx_1)
    print(f"Delay estmation: {delay*1000.0/Fs:.2f} ms")

    if(delay > 0):
        audio2 = audio2[delay:]
        audio2 = np.pad(audio2, (0, delay))
    elif(delay < 0):
        delay = -1*delay
        audio1 = audio1[delay:]
        audio1 = np.pad(audio1, (0, delay))
    return audio1, audio2

#rms function
def rms(signal):
    return np.sqrt(np.mean(signal**2))

#Select audio file
audio_dir = "..\\stereo recordings\\"

if not os.path.exists(audio_dir):
    raise FileNotFoundError(f"Directory does not exist: {os.path.abspath(audio_dir)}")

#List files
file_list = sorted(os.listdir(audio_dir))
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

print(f"File selected: {file_list[selection]}")

# audiofile = audiofile[:5*Fs]
audio_dtype = audiofile.dtype
tlen = len(audiofile)/Fs
print(f"Fs = {Fs} kHz, Duration = {tlen:.2f} s")
print(f"Type = {audiofile.dtype}")

# #example tone for debugging
# sin_freq = 2000
# tone = lr.tone(frequency = sin_freq, sr = Fs, duration = tlen)

#Split sterero recording into separate channels and normalize to range [-1.0, 1.0]
bone_rec = normalize_audio(audiofile[:,0], audio_dtype) #V2S sensor audio on left channel
air_rec = normalize_audio(audiofile[:,1], audio_dtype) #Air mic audio on right channel

#zero mean
bone_rec = bone_rec - np.mean(bone_rec)
air_rec = air_rec - np.mean(air_rec)

#align bone recording and tone 
air_rec, bone_rec = align(air_rec, bone_rec)

active_start = int(1*Fs)
active_end = int(active_start + 4*Fs)
inactive_start = int(4.5*Fs)
inactive_end = int(7.5*Fs)

mixed_rms = rms(bone_rec[active_start:active_end])
noise_rms = rms(bone_rec[inactive_start:inactive_end])
signal_rms = np.sqrt(np.maximum(mixed_rms**2 - noise_rms**2, 1e-12))
signal_rms = mixed_rms #for debugging
print(f"BCM Noise power: {20*np.log10(noise_rms):.2f} dBFS")
snrbone_db = 20*np.log10(signal_rms/noise_rms)

mixed_rms = rms(air_rec[active_start:active_end])
noise_rms = rms(air_rec[inactive_start:inactive_end])
signal_rms = np.sqrt(np.maximum(mixed_rms**2 - noise_rms**2, 1e-12))
signal_rms = mixed_rms #for debugging
print(f"ACM Noise power: {20*np.log10(noise_rms):.2f} dBFS")
snrair_db = 20*np.log10(signal_rms/noise_rms)

print(f"SNR BONE: {snrbone_db:.2f} dB")
print(f"SNR AIR: {snrair_db:.2f} dB")


