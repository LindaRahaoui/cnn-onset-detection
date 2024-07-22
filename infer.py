import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from utils import onsetCNN

def load_model(model_path, device):
    model = onsetCNN().double().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_audio(audio_path, sr=44100, n_fft=1024, hop_length=441, n_mels=80, fmin=27.5, fmax=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def predict_onsets(model, mel_spectrogram_db, device, hop_length=441, sr=44100):
    # Segment the Mel spectrogram into overlapping windows for the model
    segment_length = 15  # As used during training
    num_segments = mel_spectrogram_db.shape[1] - segment_length + 1

    # Create an empty array to store predictions
    onsets = np.zeros(mel_spectrogram_db.shape[1])

    with torch.no_grad():
        for i in range(num_segments):
            segment = mel_spectrogram_db[:, i:i+segment_length]
            segment = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
            prediction = model(segment).squeeze().cpu().numpy()
            onsets[i:i+segment_length] += prediction

    return onsets

def plot_onsets(onsets, audio_path, sr=44100, hop_length=441):
    # Plot the original audio waveform
    y, sr = librosa.load(audio_path, sr=sr)
    times = np.arange(len(y)) / sr

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(times, y, label='Waveform')
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot the onset predictions
    onset_times = np.arange(len(onsets)) * hop_length / sr
    plt.subplot(2, 1, 2)
    plt.plot(onset_times, onsets, label='Onset Predictions')
    plt.title('Onset Predictions')
    plt.xlabel('Time (s)')
    plt.ylabel('Onset Confidence')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(audio_path, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    mel_spectrogram_db = preprocess_audio(audio_path)
    onsets = predict_onsets(model, mel_spectrogram_db, device)
    plot_onsets(onsets, audio_path)

if __name__ == "__main__":
    audio_path = '../../06-L6-F#4-140708_1103.1.wav'  # Replace with the path to your WAV file
    model_path = 'models/saved_model_0_49.pt'  # Replace with the path to your trained model
    main(audio_path, model_path)
