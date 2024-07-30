import numpy as np
import os
import librosa

# Data directory
audio_dir = '/content/drive/MyDrive/Dataset_cnn/wav'

# Load the list of song names from 'songlist.txt' 
with open('songlist.txt', 'r') as file:
    songlist = file.read().splitlines()


# Initialize lists to store means and standard deviations
means_song = [np.array([]), np.array([]), np.array([])]
stds_song = [np.array([]), np.array([]), np.array([])]

for i_song in range(len(songlist)):
    # Construct the full file path and normalize it
    file_path = os.path.normpath(os.path.join(audio_dir, songlist[i_song] + '.wav'))
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    # Load audio
    x, fs = librosa.load(file_path, sr=44100)
    
    # Get mel spectrograms with different FFT sizes
    melgram1 = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=1024, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    melgram2 = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=2048, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    melgram3 = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=4096, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    
    # Log scaling
    melgram1 = 10 * np.log10(1e-10 + melgram1)
    melgram2 = 10 * np.log10(1e-10 + melgram2)
    melgram3 = 10 * np.log10(1e-10 + melgram3)
    
    # Compute mean and std of dataset
    if i_song == 0:
        means_song[0] = np.mean(melgram1, axis=1)
        means_song[1] = np.mean(melgram2, axis=1)
        means_song[2] = np.mean(melgram3, axis=1)
        
        stds_song[0] = np.std(melgram1, axis=1)
        stds_song[1] = np.std(melgram2, axis=1)
        stds_song[2] = np.std(melgram3, axis=1)
    else:
        means_song[0] += np.mean(melgram1, axis=1)
        means_song[1] += np.mean(melgram2, axis=1)
        means_song[2] += np.mean(melgram3, axis=1)
        
        stds_song[0] += np.std(melgram1, axis=1)
        stds_song[1] += np.std(melgram2, axis=1)
        stds_song[2] += np.std(melgram3, axis=1)

# Compute the final means and stds
means_song[0] /= len(songlist)
means_song[1] /= len(songlist)
means_song[2] /= len(songlist)

stds_song[0] /= len(songlist)
stds_song[1] /= len(songlist)
stds_song[2] /= len(songlist)

# Save the results
np.save('means_stds.npy', np.array([means_song, stds_song]))
