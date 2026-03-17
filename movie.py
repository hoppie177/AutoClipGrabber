import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment

# -----------------------------
# SETTINGS (TUNE THESE)
# -----------------------------
INPUT_VIDEO = "stream.mp4"
OUTPUT_PREFIX = "clip"
CLIP_DURATION = 15          # seconds per clip
THRESHOLD_MULTIPLIER = 3.5 # higher = fewer clips

# -----------------------------
# LOAD VIDEO + AUDIO
# -----------------------------
video = VideoFileClip(INPUT_VIDEO)
audio = AudioSegment.from_file(INPUT_VIDEO)

# Convert audio to numpy array
samples = np.array(audio.get_array_of_samples())

# If stereo → make mono
if audio.channels == 2:
    samples = samples.reshape((-1, 2))
    samples = samples.mean(axis=1)

# Normalize
samples = samples / np.max(np.abs(samples))

# -----------------------------
# ANALYZE AUDIO LOUDNESS
# -----------------------------
window_size = 44100  # 1 second chunks (assuming 44.1kHz)
loudness = []

for i in range(0, len(samples), window_size):
    chunk = samples[i:i+window_size]
    if len(chunk) == 0:
        continue
    volume = np.sqrt(np.mean(chunk**2))  # RMS
    loudness.append(volume)

loudness = np.array(loudness)

# -----------------------------
# FIND PEAK MOMENTS
# -----------------------------
avg_volume = np.mean(loudness)
threshold = avg_volume * THRESHOLD_MULTIPLIER

peaks = np.where(loudness > threshold)[0]

# Merge nearby peaks
selected_moments = []
min_gap = 30  # seconds between clips

for peak in peaks:
    if not selected_moments:
        selected_moments.append(peak)
    elif peak - selected_moments[-1] > min_gap:
        selected_moments.append(peak)

# -----------------------------
# CUT CLIPS
# -----------------------------
print(f"Found {len(selected_moments)} potential clips...")

for i, moment in enumerate(selected_moments):
    start = max(0, moment - CLIP_DURATION // 2)
    end = start + CLIP_DURATION

    clip = video.subclipped(start, end)
    clip.write_videofile(f"{OUTPUT_PREFIX}_{i+1}.mp4")

print("Done!")