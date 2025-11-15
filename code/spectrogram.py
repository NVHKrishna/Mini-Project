from pathlib import Path
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# Frequencies
TICKS = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 12000])
TICK_LABELS = np.array(["31.25", "62.5", "125", "250", "500", "1k", "2k", "4k", "8k", "12k"])

# File structure
PROJECT_PATH = Path.home() / "Desktop" / "mini_project"
IMG_OUTPUT_PATH = PROJECT_PATH / "img"
DATA_INPUT_PATH = PROJECT_PATH / "data"

IMG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
DATA_INPUT_PATH.mkdir(parents=True, exist_ok=True)

SAVE_PARAMS = {"dpi": 300, "bbox_inches": "tight"}

# -----------------------------
# Spectrogram function
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

    # Save image using the audio file name
    save_file = IMG_OUTPUT_PATH / f"{audio_file_path.stem}_spectrogram.png"
    plt.savefig(save_file, **SAVE_PARAMS)
    plt.close()
    print(f"✅ Saved spectrogram to {save_file}")

# -----------------------------
# Main
# -----------------------------
def main():
    plt.rcParams.update({"font.size": 18})
    
    # Example audio file
    audio_file = DATA_INPUT_PATH / "cat0004.wav"
    
    if not audio_file.is_file():
        print(f"❌ Audio file not found: {audio_file}")
        print("Place the audio file in the 'data' folder on your Desktop.")
        return

    signal, sample_rate = sf.read(audio_file)
    print(f"🎵 Loaded: {audio_file.name}, shape={signal.shape}, fs={sample_rate}")

    plot_and_save_spectrogram(signal, sample_rate, audio_file)

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    print("🚀 Script started !!")
    main()
    print("✅ Script finished !!")
