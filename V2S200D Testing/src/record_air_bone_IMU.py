import serial
import os
import sounddevice as sd
import numpy as np
import threading
import time
from scipy.io.wavfile import write
from datetime import datetime

# ==== CONFIG ====
SERIAL_PORT = "COM3"      # Replace with your Arduino port
BAUD_RATE = 115200
AUDIO_DIR = "../imuVairVbone_recordings/stereorecordings"
CSV_DIR = "../imuVairVbone_recordings/BNO055_logs"
AUDIO_SR = 48000          # Hz
CHANNELS = 2              # mono=1, stereo=2
DURATION = 10             # seconds

# ==== Arduino logger ====
def log_arduino(filename, stop_event):
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    with open(filename, "w") as f:
        f.write("time_ms,X,Y,Z\n")  # CSV header
        while not stop_event.is_set():
            try:
                line = arduino.readline().decode(errors="ignore").strip()
                if line:
                    t = int(time.time() * 1000)  # host timestamp (ms)
                    f.write(f"{t},{line}\n")
                    print("[ESP32_IMU]", line)
            except Exception as e:
                print("Serial error:", e)
                break
    arduino.close()

# ==== Microphone recorder ====
def record_audio(filename, samplerate, duration, channels, stop_event):
    print("[Audio] Recording...")
    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=channels,
                   dtype="int16")
    sd.wait()
    write(filename, samplerate, audio)
    print(f"[Audio] Saved {filename}")

# ==== Main ====
if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    audio_path = os.path.join(AUDIO_DIR, f"mic_recording_{timestamp}.wav")
    csv_path = os.path.join(CSV_DIR, f"bno055_data_{timestamp}.csv")

    stop_event = threading.Event()

    # Start Arduino logging in background
    t1 = threading.Thread(target=log_arduino, args=(csv_path, stop_event))
    t1.start()

    # Record audio (blocks until finished)
    record_audio(audio_path, AUDIO_SR, DURATION, CHANNELS, stop_event)

    stop_event.set()  # Signal Arduino thread to stop

    # Wait for Arduino thread to finish
    t1.join()
    print("All done.")
