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

print(f"\nSelected: {file_list[selection]}")

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

#zero mean for SNR and statistical analysis
bone_rec = bone_rec - np.mean(bone_rec)
air_rec = air_rec - np.mean(air_rec)

#filter out 7kHz resonance from the V2S200D mic
f0 = 7000 #set Q factor and resonant frequency of notch filter
Q_factor = 1
b, a = signal.iirnotch(w0 = f0, Q = Q_factor, fs = Fs)
bone_rec = signal.filtfilt(b, a, x = bone_rec)

#initialize parameters for VAD
frame_length = 40
frame_shift = frame_length // 4
energy_threshold = 0.01
pre_emphasis = 0.95

#instantiate vad object
vad = EnergyVAD(    sample_rate = Fs,
                    frame_length = frame_length,
                    frame_shift = frame_shift,
                    energy_threshold = energy_threshold,
                    pre_emphasis = pre_emphasis)

#create voice activity mask and inverse
voice_activity = vad(bone_rec)
voice_inactivity = 1 - voice_activity


#smooth the mask with a raised cosine window
Nx = 10
window = signal.get_window('blackman', Nx = Nx)
mask = signal.convolve(voice_activity, window, mode = 'same')
#normalize mask
mask = mask/np.max(mask)

#lp filter the voice activity array
# b, a = signal.iirdesign(0.3, 0.7, 1, 40, ftype = 'cheby1')
# voice_activity = signal.lfilter(b,a, x = voice_activity)

#initialize vars for tracking active/inactive time
active_time = 0
inactive_time = tlen*Fs

inactive_bone = np.zeros(bone_rec.shape)
inactive_air = np.zeros(bone_rec.shape)
active_air = np.zeros(bone_rec.shape)
active_bone = np.zeros(bone_rec.shape)

output = np.zeros((len(bone_rec),))
frame_length = Fs*frame_length //1000
frame_shift = Fs*frame_shift //1000

#apply VAD mask to air conduction
for i in range(voice_activity.shape[0]):
    active_time = active_time + voice_activity[i]*(frame_shift)
    inactive_time = inactive_time - voice_activity[i]*(frame_shift)
    inactive_air[i*frame_shift: i*frame_shift + frame_length] = \
        (1 - voice_activity[i])*air_rec[i*frame_shift: i*frame_shift + frame_length]
    
    inactive_bone[i*frame_shift: i*frame_shift + frame_length] = \
        (1 - voice_activity[i])*bone_rec[i*frame_shift: i*frame_shift + frame_length]
    
    active_air[i*frame_shift : i*frame_shift + frame_length] = \
        voice_activity[i]*air_rec[i*frame_shift : i*frame_shift + frame_length]
    active_bone[i*frame_shift : i*frame_shift + frame_length] = \
        voice_activity[i]*air_rec[i*frame_shift : i*frame_shift + frame_length]
    
    #output uses the smoothed mask
    output[i*frame_shift : i*frame_shift + frame_length] = \
        mask[i]*air_rec[i*frame_shift : i*frame_shift + frame_length]

print(f"Voice active time: {1000*active_time/Fs} ms")
print(f"Voice inactive time: {1000*inactive_time/Fs:.1f} ms")

#SNR ESTIMATION OF SPEECH OR COUGHING RECORDING

#isolate all nonzero samples for active and inactive components

#concatenate isolated segments
    # for noise, sort segments by length before concatenation

#truncate the longer segment to equal the length of the shorter segment

#compute RMS amplitude speech and noise 

#compute average SNR estimate 

#ALTERNATIVE: output concatenated files and compute snr in another script

#rewrite air recording with vad and save to new stereo file
outfile = np.zeros(audiofile.shape).astype(np.float32)
outfile[:,0] = bone_rec.astype(np.float32)
outfile[:,1] = output.astype(np.float32)
audio_dir = "../stereo recordings"
outfilename = f"VAD_{file_list[selection]}"
write(os.path.join(audio_dir, outfilename), Fs, outfile)

fig, axes = plt.subplots(3,1)

plt.figure(1)
t = np.linspace(0,tlen,int(48000*tlen))
axes[0].plot(t,air_rec)
axes[0].plot(t, air_rec, alpha=0.5)
axes[0].fill_between(t, -1, 1, where=(output != 0), color='orange', alpha=0.2, label="VAD Active")
axes[0].legend()
n = np.linspace(0,len(voice_activity)/100,len(voice_activity))
axes[1].plot(n,mask)
axes[2].plot(t,output)
axes[2].set_ylim((-1,1))

fig2, axes2 = plt.subplots(2,1)
plt.figure(2)
axes2[0].plot(t,inactive_air)
axes2[0].set_title("Air Recording Noise")
axes2[0].set_ylim((-1,1))
axes2[0].fill_between(t, -1, 1, where=(output != 0), color='orange', alpha=0.2, label="VAD Active")
axes2[0].legend()
axes2[1].plot(t,inactive_bone)
axes2[1].set_title("Bone Recording Noise")
axes2[1].set_ylim((-1,1))
axes2[1].fill_between(t, -1, 1, where=(output != 0), color='orange', alpha=0.2, label="VAD Active")
axes2[1].legend()
plt.show()



