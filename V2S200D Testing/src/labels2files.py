"""
DISCLAIMER: THIS WAS WRITTEN MOSTLY BY CHATGPT
"""

import os
from scipy.io import wavfile

# === CONFIG ===
label_file = "clavicle labels.txt"           # Your label file
label_dir = "../labels/7-18-25 preliminary BCM placement testing/raw_labels"
audio_dir = "H:\My Drive\ARoS Lab\stereo recordings"
audio_file = "7-18-25 BCM clavicle placement fullrecording.wav"       # Your audio file
output_dir = "H:\My Drive\ARoS Lab\stereo recordings"             # Output folder

# === LOAD AUDIO ===
sr, audio = wavfile.read(os.path.join(audio_dir,audio_file))

# === LOAD LABELS ===
labels = []
with open(os.path.join(label_dir, label_file), 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3:
            continue  # skip malformed lines
        start, end, label = parts
        labels.append({
            "start": float(start),
            "end": float(end),
            "label": label
        })

# === PROCESS SEGMENTS ===
for i, item in enumerate(labels):
    start_sample = int(float(item["start"]) * sr)
    end_sample = int(float(item["end"]) * sr)
    segment = audio[start_sample:end_sample]

    # Sanitize filename
    label_name = ''.join(c if c.isalnum() else '_' for c in item["label"])
    filename = f"{"7-18-25_BCM_placement_test"}_{label_name}.wav"
    filepath = os.path.join(output_dir, filename)

    wavfile.write(filepath, sr, segment.astype(audio.dtype))
    print(f"Saved: {filepath}")
