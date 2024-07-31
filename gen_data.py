import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import librosa
import torch

# Function to zero pad ends of spectrogram
def zeropad2d(x, n_frames):
    y = np.hstack((np.zeros([x.shape[0], n_frames]), x))
    y = np.hstack((y, np.zeros([x.shape[0], n_frames])))
    return y

# Function to create N-frame overlapping chunks of the full audio spectrogram  
def makechunks(x, duration):
    y = np.zeros([x.shape[1], x.shape[0], duration])
    for i_frame in range(x.shape[1] - duration):
        y[i_frame] = x[:, i_frame:i_frame + duration]
    return y

# Data dirs
audio_dir = '/content/drive/MyDrive/Dataset_cnn/wav'
onset_dir = '/content/drive/MyDrive/Dataset_cnn/annotations/'
save_dir = '/content/drive/MyDrive/Dataset_cnn/data_pt_test'

# Data stats for normalization
stats = np.load('/content/drive/MyDrive/Dataset_cnn/means_stds.npy')
means = stats[0]
stds = stats[1]

# Context parameters
contextlen = 7  # +- frames
duration = 2 * contextlen + 1

# Main
with open('/content/cnn-onset-detection/songlist.txt', 'r') as file:
    songlist = file.read().splitlines()
audio_format = '.wav'
labels_master = {}
weights_master = {}
filelist = []
total_files = len(songlist)
processed_files = 0

# Count already processed files
for item in songlist:
    savedir = os.path.join(save_dir, item)
    if os.path.exists(savedir) and len(os.listdir(savedir)) > 1005:
        processed_files += 1

for item in songlist:
    savedir = os.path.join(save_dir, item)
    
    # Skip already processed items if there are more than 1000 files
    if os.path.exists(savedir) and len(os.listdir(savedir)) > 1005:
        print(f'Skipping {item}, already processed.')
        continue
    
    # Load audio and onsets
    x, fs = librosa.load(os.path.join(audio_dir, item + audio_format), sr=44100)
    print(onset_dir, item + '_annotation.csv')
    if not os.path.exists(os.path.join(onset_dir, item + '_annotation.csv')):
        path = os.path.join(onset_dir, item + '_annotation.csv')
        print(f"{path} n'existe pas")
        break
    
    # Load only the first column (onsets) and skip the first row (header)
    onsets = np.loadtxt(os.path.join(onset_dir, item + '_annotation.csv'), delimiter=',', skiprows=1, usecols=0)
    
    # Get mel spectrogram
    melgram1 = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=1024, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    melgram2 = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=2048, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    melgram3 = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=4096, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    
    # Log scaling
    melgram1 = 10 * np.log10(1e-10 + melgram1)
    melgram2 = 10 * np.log10(1e-10 + melgram2)
    melgram3 = 10 * np.log10(1e-10 + melgram3)
    
    # Normalize
    melgram1 = (melgram1 - np.atleast_2d(means[0]).T) / np.atleast_2d(stds[0]).T
    melgram2 = (melgram2 - np.atleast_2d(means[1]).T) / np.atleast_2d(stds[1]).T
    melgram3 = (melgram3 - np.atleast_2d(means[2]).T) / np.atleast_2d(stds[2]).T
    
    # Zero pad ends
    melgram1 = zeropad2d(melgram1, contextlen)
    melgram2 = zeropad2d(melgram2, contextlen)
    melgram3 = zeropad2d(melgram3, contextlen)
    
    # Make chunks
    melgram1_chunks = makechunks(melgram1, duration)
    melgram2_chunks = makechunks(melgram2, duration)
    melgram3_chunks = makechunks(melgram3, duration)
    
    # Generate song labels
    hop_dur = 10e-3
    labels = np.zeros(melgram1_chunks.shape[0])
    weights = np.ones(melgram1_chunks.shape[0])
    idxs = np.array(np.round(onsets / hop_dur), dtype=int)
    labels[idxs] = 1
    
    # Target smearing
    labels[idxs - 1] = 1
    labels[idxs + 1] = 1
    weights[idxs - 1] = 0.25
    weights[idxs + 1] = 0.25

    labels_dict = {}
    weights_dict = {}
    
    # Save
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    for i_chunk in range(melgram1_chunks.shape[0]):
        savepath = os.path.join(savedir, str(i_chunk) + '.pt')
        torch.save(torch.tensor(np.array([melgram1_chunks[i_chunk], melgram2_chunks[i_chunk], melgram3_chunks[i_chunk]])), savepath)
        
        # Verify if the file was created
        if not os.path.exists(savepath):
            print(f'Error: {savepath} was not created.')
            break
            
        filelist.append(savepath)
        labels_dict[savepath] = labels[i_chunk]
        weights_dict[savepath] = weights[i_chunk]

    # Append labels to master
    labels_master.update(labels_dict)
    weights_master.update(weights_dict)
    
    np.savetxt(os.path.join(savedir, 'labels.txt'), labels)
    np.savetxt(os.path.join(savedir, 'weights.txt'), weights)
    
    processed_files += 1
    print(f'Processed {processed_files}/{total_files} files.')

np.save('labels_master.npy', labels_master)
np.save('weights_master.npy', weights_master)
np.savetxt('filelist.txt', filelist, fmt='%s')
