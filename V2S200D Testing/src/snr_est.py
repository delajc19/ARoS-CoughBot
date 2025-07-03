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
import matplotlib.pyplot as plt
import csv

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
def align(audio1, audio2, Fs):
    Fs = Fs
    Rxx_1 = signal.correlate(audio1, audio1, mode = 'full')
    Rxy_2 = signal.correlate(audio2, audio1, mode = 'full')
    delay = np.argmax(Rxy_2) - np.argmax(Rxx_1)

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

#calculate snr for the bone and air conduction recordings of a given file
#returns: snr bone, snr air, corr coef bone, corr coef air
def calculate_snr(filename):
    # audiofile = audiofile[:5*Fs]
    Fs, audiofile = read(filename)
    audio_dtype = audiofile.dtype
    tlen = len(audiofile)/Fs
    # print(f"Fs = {Fs} kHz, Duration = {tlen:.2f} s")
    # print(f"Type = {audiofile.dtype}")
    
    #Split sterero recording into separate channels and normalize to range [-1.0, 1.0]
    bone_rec = normalize_audio(audiofile[:,0], audio_dtype) #V2S sensor audio on left channel
    air_rec = normalize_audio(audiofile[:,1], audio_dtype) #Air mic audio on right channel

    #zero mean
    bone_rec = bone_rec - np.mean(bone_rec)
    air_rec = air_rec - np.mean(air_rec)

    #align bone recording and tone 
    air_rec, bone_rec = align(air_rec, bone_rec, Fs)

    active_start = int(1*Fs)
    active_end = int(4*Fs)
    inactive_start = int(4.5*Fs)
    inactive_end = int(7.5*Fs)

    active_seg = bone_rec[active_start:active_end]
    inactive_seg = bone_rec[inactive_start:inactive_end]

    corr_bone = np.corrcoef(active_seg, inactive_seg)[0, 1]

    mixed_rms = rms(active_seg)
    noise_rms = rms(inactive_seg)
    signal_rms = np.sqrt(np.maximum(mixed_rms**2 - noise_rms**2, 0))
    snrbone_db = 20*np.log10(signal_rms/noise_rms)

    active_seg = air_rec[active_start:active_end]
    inactive_seg = air_rec[inactive_start:inactive_end]

    corr_air = np.corrcoef(active_seg, inactive_seg)[0, 1]

    mixed_rms = rms(active_seg)
    noise_rms = rms(inactive_seg)
    signal_rms = np.sqrt(np.maximum(mixed_rms**2 - noise_rms**2, 0))
    snrair_db = 20*np.log10(signal_rms/noise_rms)

    return snrbone_db, snrair_db, corr_bone, corr_air


    

#Select audio file
audio_dir = "..\\stereo recordings\\"

if not os.path.exists(audio_dir):
    raise FileNotFoundError(f"Directory does not exist: {os.path.abspath(audio_dir)}")

#List files
file_list = sorted(os.listdir(audio_dir))
num_files = len(file_list)

snrvals = np.zeros((3,8,2), float)
corrvals = np.zeros((3,8,2), float)
frequencies = [10000, 1000, 100, 2000, 3000, 5000, 500, 7000]
# snrvals[3,:,1] = frequencies 
# corrvals[3,:,1] = frequencies 


j = 0
for i in range(num_files):
    noise_idx = -1
    if "HN_ST" in file_list[i]:
        noise_idx = 0
    if "LN_ST" in file_list[i]:
        noise_idx = 1
    if "Q_ST" in file_list[i]:
        noise_idx = 2
    
    filename = os.path.join(audio_dir, file_list[i])
    if(noise_idx >= 0):
        if j > 7: j = 0
        snr_bone, snr_air, corr_bone, corr_air = calculate_snr(filename)
        snrvals[noise_idx,j,0] = snr_bone
        snrvals[noise_idx,j,1] = snr_air
        corrvals[noise_idx,j,0] = corr_bone
        corrvals[noise_idx,j,1] = corr_air
        j += 1

#Write values to .csv
with open("../csv files/snrvals.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    row = ['','','','','','','','']

    for i in range(len(frequencies)):
        row[i] = f"{frequencies[i]:d} [Hz]"
    writer.writerow(row)

    for i in range(snrvals.shape[0]):
        for j in range(snrvals.shape[1]):
            row[j] = f"{snrvals[i, j, 0]:.2f}, {snrvals[i, j, 1]:.2f}"
        writer.writerow(row)

with open("../csv files/corrvals.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    row = ['','','','','','','','']

    for i in range(len(frequencies)):
        row[i] = f"{frequencies[i]:d} [Hz]"
    writer.writerow(row)  

    for i in range(corrvals.shape[0]):
        for j in range(corrvals.shape[1]):
            row[j] = f"{corrvals[i, j, 0]:.2f}, {corrvals[i, j, 1]:.2f}"
        writer.writerow(row)



plt.show()
