import sounddevice as sd
from scipy.io.wavfile import write


#Recording setup
Fs = 44100 #44.1 kHz sample rate
duration = 10 #10 second recording
input_device = 1 #Index of USB V2S-Demo in device query
output_device = 6
channels = 2 #Stereo recording

#Record
recording = sd.rec(
    int(Fs*duration), 
    samplerate = Fs, 
    channels = channels, 
    dtype = "float32", 
    device = (input_device, output_device))
sd.wait()
write("test_recording.wav", Fs, recording)
print("Recording completed")




    
    





    