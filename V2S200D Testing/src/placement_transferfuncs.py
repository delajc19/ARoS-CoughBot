import os
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy import signal
from itertools import combinations, permutations
import tkinter as tk
from tkinter import filedialog


MAX_INT16 = 32767.0
MAX_INT32 = 2147483647.0

def normalize_audio(audio, dtype):
    if dtype == "float32":
        return np.array(audio).astype(np.float32)
    elif dtype == "int16":
        return audio / MAX_INT16
    elif dtype == "int32":
        return audio / MAX_INT32
    else:
        raise ValueError("Unsupported audio dtype")

def butter_bp(lowcut, highcut, fs, order):
    w1 = 2*lowcut / fs
    w2 = 2*highcut/ fs
    b, a = signal.butter(order,[w1, w2], btype = 'bandpass', analog = False)
    return b, a
    
def compute_transfer_function(bone_x, bone_y, Fs, n_window=8092, hop_ratio=4):
    window = 'hann'
    hop_length = n_window // hop_ratio

    # Ensure both signals are the same length
    min_len = min(len(bone_x), len(bone_y))
    bone_x = bone_x[:min_len]
    bone_y = bone_y[:min_len]

    #Filter both to range 60-3000Hz
    b,a = butter_bp(60,3000,Fs,order = 5)
    bone_x = signal.filtfilt(b, a, bone_x)
    bone_y = signal.filtfilt(b, a, bone_y)

    # STFT
    Sxx_x = lr.stft(bone_x, n_fft=n_window, hop_length=hop_length, window=window)
    Sxx_y = lr.stft(bone_y, n_fft=n_window, hop_length=hop_length, window=window)
    H_stft = np.abs(Sxx_x / (Sxx_y + 1e-10))
    H_avg = np.mean(H_stft, axis=1)

    #average
    kernel = np.ones((n_window//64,))
    H_avg = np.convolve(a=H_avg, v=kernel,mode='same')
    H_avg_dB = 10 * np.log10(H_avg / np.max(H_avg))

    freqs = lr.fft_frequencies(sr=Fs, n_fft=n_window)
    return freqs, H_avg_dB

def truncate_to_same_length(signals):
    """
    Truncates a list of 1D numpy arrays to the length of the shortest array.

    Parameters:
        signals (List[np.ndarray]): List of 1D arrays (e.g., audio signals)

    Returns:
        List[np.ndarray]: Truncated arrays, all same length
    """
    if not signals:
        return []

    min_len = min(len(sig) for sig in signals)
    return [sig[:min_len] for sig in signals]

def select_audio_files(n=4, title="Select 4 Stereo Audio Files"):
    """
    Open a file dialog to select `n` audio files in order.
    
    Returns:
        List of file paths in the selected order.
    """
    root = tk.Tk()
    root.withdraw()  # hide the main window
    file_paths = filedialog.askopenfilenames(
        title=title,
        filetypes=[("WAV files", "*.wav")],
        multiple=True
    )
    root.destroy()

    file_paths = list(file_paths)
    if len(file_paths) != n:
        raise ValueError(f"Expected {n} files, but got {len(file_paths)}")
    return file_paths

#----------------------------------------------------------------------------------------
audio_dir = "H:\My Drive\ARoS Lab\stereo recordings"  # folder with stereo files
files = select_audio_files()
Fs_list, stereo_data = zip(*[read(f) for f in files])
Fs = Fs_list[0]

for i in range(len(files)):
    print(f"{i}\t{files[i]}")

if len(files) < 4:
    raise ValueError("Please provide at least 4 stereo WAV files.")

air_mics = []
bone_mics = []

#filter out 7kHz resonance from the V2S200D mic
f0 = 7000 #set Q factor and resonant frequency of notch filter
Q_factor = 1
b, a = signal.iirnotch(w0 = f0, Q = Q_factor, fs = Fs)
alpha = 0.7 #filter influence
# for i in range(len(bone_mics)):
#     bone_mics[i] = alpha*signal.filtfilt(b, a, x = bone_mics[i]) + (1-alpha)*bone_mics[i]


air_mics = [x[:, 0] for x in stereo_data]
bone_mics = [x[:, 1] for x in stereo_data]
    
bone_mics = truncate_to_same_length(bone_mics)
air_mics = truncate_to_same_length(air_mics)


# Generate all unique pairwise combinations (i ≠ j)
pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
names = ['chest', 'clavicle', 'head', 'throat']
truncate_to_same_length(bone_mics)

#set up averaging
kernel = np.ones((5,))

# Create plots
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

# One subplot per numerator
for numerator_idx in range(4):
    ax = axes[numerator_idx]
    ax.set_title(f"Transfer Functions: BCM {names[numerator_idx]}")
    ax.set_ylabel("Gain [dB]")
    ax.grid(True)

    for denominator_idx in range(4):
        if denominator_idx == numerator_idx:
            continue  # skip self

        # Compute TF
        freqs, H_avg_dB = compute_transfer_function(
            bone_mics[numerator_idx],
            bone_mics[denominator_idx],
            Fs
        )

        # Plot
        label = f"BCM {names[numerator_idx]} / {names[denominator_idx]}"
        ax.plot(freqs, H_avg_dB, label=label)

    ax.legend()

axes[-1].set_xlabel("Frequency [Hz]")
fig.suptitle("Bone Mic Placement Transfer Functions", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])

# Create plots
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

for numerator_idx in range(4):
    ax = axes[numerator_idx]
    ax.set_title(f"Transfer Functions: ACM {names[numerator_idx]}")
    ax.set_ylabel("Gain [dB]")
    ax.grid(True)

    for denominator_idx in range(4):
        if denominator_idx == numerator_idx:
            continue  # skip self

        # Compute TF
        freqs, H_avg_dB = compute_transfer_function(
            air_mics[numerator_idx],
            air_mics[denominator_idx],
            Fs
        )

        # Plot
        label = f"ACM {names[numerator_idx]} / {names[denominator_idx]}"
        ax.plot(freqs, H_avg_dB, label=label)

    ax.legend()

axes[-1].set_xlabel("Frequency [Hz]")
# axes[-1].set_xlim([0,2000])
# axes[-1].set_ylim([-50,0])
fig.suptitle("Air Mic Placement Transfer Functions", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

