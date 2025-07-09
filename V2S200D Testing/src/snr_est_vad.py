#Joseph de la Viesca

import os
import numpy as np
from scipy.io.wavfile import read, write
import scipy.signal as signal
import matplotlib.pyplot as plt
from vad import EnergyVAD
import sounddevice as sd

#Initialize constants
MAX_INT16 = 32767.0
MAX_INT32 = 2147483647.0

#convert audio samples to float32 and normalize to range [-1,1]
def normalize_audio(audio, dtype):
    if(dtype=="float32"):
        return np.array(audio).astype(np.float32)
    elif(dtype=="int16"):
        return audio/MAX_INT16 #Normalize by max int16
    elif(dtype=="int32"):
        return audio/MAX_INT32 #Normalize by max int32

#identify segments of contiguous active (nonzero) audio data and store in a list
def isolate_activity(audiofile):
    #isolate all nonzero samples for active and inactive components
    segments = []
    startseg = 0
    endseg = 0
    sidx = 0
    for i in range(len(audiofile)):
        if(i > 0 and audiofile[i]!=0 and audiofile[i-1]==0):
            startseg = i

        if(i > 0 and audiofile[i] == 0 and audiofile[i-1] != 0):
            endseg = i
            segments.append(audiofile[startseg:endseg])
            sidx = 1 + sidx
    
    return segments

#concatenate isolated segments
def stitch(input_list):
    stitched = np.array(input_list[0])
    for i in range(len(input_list)-1):
        stitched = np.append(stitched, input_list[i+1])
    return stitched

#sort list by length (longest to shortest)
def sort_by_len(input_list):
    for i in range(len(input_list)):
        temp_list = []
        for j in range(len(input_list)):
            if(len(input_list[j])>len(input_list[i])):
                temp_list = input_list[i]
                input_list[i] = input_list[j]
                input_list[j] = temp_list
    return input_list

#rms function
def rms(signal):
    return np.sqrt(np.mean(signal**2))

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
window = signal.get_window('hann', Nx = Nx)
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
        voice_activity[i]*bone_rec[i*frame_shift : i*frame_shift + frame_length]
    
    #output uses the smoothed mask
    output[i*frame_shift : i*frame_shift + frame_length] = \
        mask[i]*air_rec[i*frame_shift : i*frame_shift + frame_length]

print(f"Voice active time: {1000*active_time/Fs} ms")
print(f"Voice inactive time: {1000*inactive_time/Fs:.1f} ms")

#SNR ESTIMATION OF SPEECH OR COUGHING RECORDING

segmented_airspeech = isolate_activity(active_air)
segmented_bonespeech = isolate_activity(active_bone)
segmented_airnoise = isolate_activity(inactive_air)
segmented_bonenoise = isolate_activity(inactive_bone)

#sort noise segments by length before concatenation 
segmented_airnoise = sort_by_len(segmented_airnoise)
segmented_bonenoise = sort_by_len(segmented_bonenoise)

#concatenate segments into one audio file
stitched_airspeech = stitch(segmented_airspeech)
stitched_bonespeech = stitch(segmented_bonespeech)
stitched_airnoise = stitch(segmented_airnoise)
stitched_bonenoise = stitch(segmented_bonenoise)

#truncate the longer segment to equal the length of the shorter segment
minlen = min(len(stitched_airnoise),len(stitched_airspeech))
stitched_airspeech = stitched_airspeech[0:minlen]
stitched_airnoise = stitched_airnoise[0:minlen]
stitched_bonespeech = stitched_bonespeech[0:minlen]
stitched_bonenoise = stitched_bonenoise[0:minlen]

#compute RMS amplitude speech and noise 
rms_airmix = rms(stitched_airspeech)
rms_airnoise = rms(stitched_airnoise)
rms_airspeech = np.sqrt(np.maximum(rms_airmix**2 - rms_airnoise**2, 0))

rms_bonemix = rms(stitched_bonespeech)
rms_bonenoise = rms(stitched_bonenoise)
rms_bonespeech = np.sqrt(np.maximum(rms_bonemix**2 - rms_bonenoise**2, 0))

#compute average SNR estimate 
SNR_air = 20*np.log10(rms_airspeech/rms_airnoise)
SNR_bone = 20*np.log10(rms_bonespeech/rms_bonenoise)

#ALTERNATIVE: output concatenated files and compute snr in another script
print(f"SNR air = {SNR_air:.2f}")
print(f"SNR bone = {SNR_bone:.2f}")

#rewrite air recording with vad and save to new stereo file
outfile = np.zeros(audiofile.shape).astype(np.float32)
outfile[:,0] = bone_rec.astype(np.float32)
outfile[:,1] = output.astype(np.float32)
audio_dir = "../stereo recordings"
outfilename = f"VAD_{file_list[selection]}"
write(os.path.join(audio_dir, outfilename), Fs, outfile)


# %% Plots
fig, axes = plt.subplots(3,1,sharex = True)

plt.figure(1)
fig.subplots_adjust(hspace = 0.5)
t = np.linspace(0,tlen,int(48000*tlen))
axes[0].plot(t,air_rec)
axes[0].plot(t, air_rec, alpha=0.5)
axes[0].fill_between(t, -1, 1, where=(active_air != 0), color='orange', alpha=0.2, label="VAD Active")
axes[0].legend()
axes[0].set_title('Input ACM Waveform')
n = np.linspace(0,len(voice_activity)/100,len(voice_activity))
axes[1].plot(n,mask)
axes[1].set_title('VAD Mask from BCM data')
axes[2].plot(t,output)
axes[2].set_title('Output ACM Waveform')
axes[2].set_ylim((-1,1))

fig2, axes2 = plt.subplots(2,1, sharex = True)
plt.figure(2)
axes2[0].plot(t,inactive_air)
axes2[0].set_title("Air Recording Noise")
axes2[0].set_ylim((-1,1))
axes2[0].fill_between(t, -1, 1, where=(active_air != 0), color='orange', alpha=0.2, label="VAD Active")
axes2[0].legend()
axes2[1].plot(t,inactive_bone)
axes2[1].set_title("Bone Recording Noise")
axes2[1].set_ylim((-1,1))
axes2[1].fill_between(t, -1, 1, where=(active_air != 0), color='orange', alpha=0.2, label="VAD Active")
axes2[1].legend()

plt.figure(3)
nst = np.linspace(0,len(stitched_airspeech)/Fs, len(stitched_airspeech))
plt.plot(nst, stitched_airspeech)


plt.show()

sd.play(stitched_airspeech, Fs)
