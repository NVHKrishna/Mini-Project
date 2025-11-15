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

# file structure
PROJECT_PATH = Path.home() / "Desktop" / "mini_project"
IMG_OUTPUT_PATH = PROJECT_PATH / "img"
DATA_INPUT_PATH = PROJECT_PATH / "data"

IMG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
DATA_INPUT_PATH.mkdir(parents=True, exist_ok=True)

SAVE_PARAMS = {"dpi": 300, "bbox_inches": "tight"}

audio_path = DATA_INPUT_PATH / "lion_roar.wav"
# name="lion"

# FUNCTION
def plot_and_save_spectrogram(signal, fs, filename=f"{audio_path}-spectrogram.", fft_size=2048, hop_size=None, window_size=None):
    
    # Generates and saves a spectrogram of an audio signal.
    if not window_size:
        window_size = fft_size
    if not hop_size:
        hop_size = window_size // 4

    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    stft = librosa.stft(signal, n_fft=fft_size, hop_length=hop_size, win_length=window_size, center=False)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
  
#   plotting
    plt.figure(figsize=(12, 6))
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
    plt.title(f"{audio_path}")
    
    output_file = IMG_OUTPUT_PATH / filename
    plt.savefig(output_file, **SAVE_PARAMS)
    print(f"Saved spectrogram to {output_file}")
    plt.close()

def main():
    plt.rcParams.update({"font.size": 20})
    
    
    #  ERROR HANDLING
    if not audio_path.is_file():
        print(f"Error: Audio file not found at {audio_path}")
        print("Please place 'sample_dog.wav' in the 'data' folder on your Desktop.")
        return
        
    signal, sample_rate = sf.read(audio_path)
    print(f"Loaded: {audio_path}, shape={signal.shape}, fs={sample_rate}")

    # FUNCTION CALL
    plot_and_save_spectrogram(signal, sample_rate, filename="spectrogram.png")

if __name__ == "__main__":
    print("Script started !! ")
    main()
    print("Script finished !! ")