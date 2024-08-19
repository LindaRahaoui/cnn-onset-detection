import numpy as np
from sklearn.model_selection import KFold
import os

# Charger la liste des chansons
with open('songlist.txt', 'r') as file:
    songlist = np.array(file.read().splitlines())

# Créer un répertoire pour les splits s'il n'existe pas
split_dir = 'dataset/splits'
if not os.path.exists(split_dir):
    os.makedirs(split_dir)

# Initialiser KFold
kf = KFold(n_splits=8, shuffle=True, random_state=42)

# Générer les splits et les enregistrer
for fold, (train_index, val_index) in enumerate(kf.split(songlist)):

    train_songs, val_songs = songlist[train_index], songlist[val_index]
    np.savetxt(os.path.join(split_dir, f'8-fold_cv_random_{fold}.fold'), val_songs, fmt='%s')
    print(f'Saved split {fold} with {len(val_songs)} validation songs.')
