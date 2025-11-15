from pathlib import Path
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# Frequencies for ticks
TICKS = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 12000])
TICK_LABELS = np.array(["31.25", "62.5", "125", "250", "500", "1k", "2k", "4k", "8k", "12k"])

# Paths
PROJECT_PATH = Path.home() / "Desktop" / "mini_project"
DATA_INPUT_PATH = PROJECT_PATH / "data"       # Folder containing all audio files
IMG_OUTPUT_PATH = PROJECT_PATH / "spectros"   # Folder to save spectrograms

# Create folder if not exists
IMG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

SAVE_PARAMS = {"dpi": 300, "bbox_inches": "tight"}

# -----------------------------
# Function to generate spectrogram
# -----------------------------
def plot_and_save_spectrogram(signal, fs, audio_file_path, fft_size=2048, hop_size=None, window_size=None):
    if not window_size:
        window_size = fft_size
    if not hop_size:
        hop_size = window_size // 4

    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    stft = librosa.stft(signal, n_fft=fft_size, hop_length=hop_size, win_length=window_size, center=False)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # White background
    plt.figure(figsize=(12, 6), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    img = librosa.display.specshow(
        spectrogram_db,
        y_axis="log",
        x_axis="time",
        sr=fs,
        hop_length=hop_size,
        cmap="inferno",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.yticks(TICKS, TICK_LABELS)
    plt.colorbar(img, format="%+2.f dBFS")
    plt.title(f"Spectrogram: {audio_file_path.stem}")

    # Save image with same name as audio file
    save_file = IMG_OUTPUT_PATH / f"{audio_file_path.stem}_spectrogram.png"
    plt.savefig(save_file, **SAVE_PARAMS)
    plt.close()
    print(f"✅ Saved spectrogram: {save_file}")

# -----------------------------
# Main
# -----------------------------
def main():
    plt.rcParams.update({"font.size": 18})

    # Scan data folder for all .wav files
    audio_files = list(DATA_INPUT_PATH.glob("*.wav"))

    if not audio_files:
        print(f"❌ No .wav files found in {DATA_INPUT_PATH}")
        return

    print(f"🎵 Found {len(audio_files)} audio files. Generating spectrograms...")

    for idx, audio_file in enumerate(audio_files, start=1):
        try:
            signal, sample_rate = sf.read(audio_file)
            print(f"Processing ({idx}/{len(audio_files)}): {audio_file.name}, shape={signal.shape}, fs={sample_rate}")
            plot_and_save_spectrogram(signal, sample_rate, audio_file)
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {type(e).__name__}: {e}")

    print("\n🏁 All spectrograms generated successfully!")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    print("🚀 Script started !!")
    main()
    print("✅ Script finished !!")
