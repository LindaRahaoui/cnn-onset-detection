import torch
from torch.utils import data
from utils import onsetCNN, Dataset
import numpy as np
import os
import glob

def main():
    # Use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Data directory
    datadir = 'dataset/data_pt'
    model_dir = '/content/drive/MyDrive/Train_New_Dataset_test2/Models'
    model_path = os.path.join(model_dir, 'model_fold_1.pt')  # Charger le modèle du premier fold par exemple
    songlist_path = 'songlist.txt'
    
    # Load the model
    model = onsetCNN().float().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test data
    with open(songlist_path, 'r') as file:
        songlist = file.read().splitlines()
    
    labels = np.load('labels_master.npy', allow_pickle=True).item()
    weights = np.load('weights_master.npy', allow_pickle=True).item()

    # Load test partition
    test_partition = []
    for song in songlist:
        folder_path = os.path.join(datadir, song)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
        ids = glob.glob(os.path.join(folder_path, '*.pt'))
        test_partition.extend(ids)
    
    test_set = Dataset(test_partition, labels, weights)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=6)
    
    # Initialize test metrics
    total_test = 0
    test_correct = 0
    test_loss = 0
    
    criterion = torch.nn.BCELoss(reduction='none')
    
    print("Testing...")
    with torch.no_grad():
        for local_batch, local_labels, local_weights in test_loader:
            total_test += local_batch.shape[0]
            local_batch, local_labels, local_weights = local_batch.to(device).float(), local_labels.to(device).float(), local_weights.to(device).float()
            
            outs = model(local_batch).squeeze()
            loss = criterion(outs, local_labels)
            loss = torch.dot(loss, local_weights)
            loss /= local_batch.size()[0]
            test_loss += loss.item()
            
            predicted = (outs > 0.5).float()
            test_correct += (predicted == local_labels).sum().item()
    
    test_loss /= total_test
    test_accuracy = test_correct / total_test
    
    print(f'Test Loss: {test_loss:.6f}\tTest Accuracy: {test_accuracy:.6f}')

if __name__ == '__main__':
    main()
