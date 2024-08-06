import os

# Définir le chemin du répertoire
directory_path = 'Small_Dataset/inference'
songlist = []

# Parcourir les fichiers dans le répertoire
for file in os.listdir(directory_path):
    # Vérifier si le fichier a l'extension .wav
    if file.endswith('.wav'):
        # Retirer l'extension et ajouter le nom du fichier à la liste
        filename_without_extension = os.path.splitext(file)[0]
        songlist.append(filename_without_extension)

# Enregistrer la liste des noms de fichiers dans un fichier texte
with open('songlist_inference.txt', 'w') as f:
    for song in songlist:
        f.write("%s\n" % song)
