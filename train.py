import sys
import glob
import torch
from torch.utils import data
from utils import onsetCNN, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to print GPU utilization
def print_gpu_utilization():
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    print(f"Allocated GPU memory: {allocated:.2f} MB")
    print(f"Cached GPU memory: {cached:.2f} MB")

# Function to repeat positive samples to improve data balance
def balance_data(ids, labels):
    ids2add = []
    for idi in ids:
        if labels[idi] == 1:
            ids2add.append(idi)
            ids2add.append(idi)
            ids2add.append(idi)
    ids.extend(ids2add)
    return ids

def main(folds):
    # Use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Parameters for data loader
    params = {'batch_size': 256, 'shuffle': True, 'num_workers': 6}
    max_epochs = 20
    patience = 5  # Patience for early stopping
    delta = 0.01  # Minimum change to qualify as an improvement

    # Data
    datadir = 'dataset/data_pt'
    os.makedirs('/content/drive/MyDrive/Train_New_Dataset_test2/Models', exist_ok=True)

    with open('songlist.txt', 'r') as file:
        songlist = file.read().splitlines()
    labels = np.load('labels_master.npy', allow_pickle=True).item()
    weights = np.load('weights_master.npy', allow_pickle=True).item()

    for fold in range(folds):
        print(f"Starting training for fold {fold + 1}/{folds}")

        # Model
        model = onsetCNN().float().to(device)
        criterion = torch.nn.BCELoss(reduction='none')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.45)

        # Cross-validation loop
        partition = {'all': [], 'train': [], 'validation': []}

        dataset_folder = 'dataset/splits'
        file_name = '8-fold_cv_random_%d.fold' % fold
        file_path = os.path.join(dataset_folder, file_name)

        # Lire le fichier ligne par ligne
        with open(file_path, 'r') as file:
            val_split = file.readlines()

        # Supprimer les caractères de fin de ligne et afficher le contenu
        val_split = [line.strip() for line in val_split]

        for song in songlist:
            folder_path = os.path.join(datadir, song)
            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
            ids = glob.glob(os.path.join(folder_path, '*.pt'))

            if song in val_split:
                partition['validation'].extend(ids)
            else:
                partition['train'].extend(ids)

        # Balance data
        partition['train'] = balance_data(partition['train'], labels)

        # Generators
        training_set = Dataset(partition['train'], labels, weights)
        training_generator = data.DataLoader(training_set, **params)

        validation_set = Dataset(partition['validation'], labels, weights)
        validation_generator = data.DataLoader(validation_set, **params)
        print("Generator done")

        # Initialize training variables
        train_loss_epoch = []
        val_loss_epoch = []
        train_acc_epoch = []
        val_acc_epoch = []

        best_val_loss = float('inf')
        early_stop_count = 0

        print("Training...")
        for epoch in range(max_epochs):
            train_loss_epoch.append(0)
            val_loss_epoch.append(0)
            train_correct = 0
            val_correct = 0
            total_train = 0
            total_val = 0

            ## GPU Utilization before epoch
            # print(f"Epoch {epoch + 1}/{max_epochs} GPU Utilization Before Training:")
            print_gpu_utilization()

            ## Training
            model.train()
            for local_batch, local_labels, local_weights in training_generator:
                total_train += local_batch.shape[0]

                # Transfer to GPU and convert to float
                local_batch, local_labels, local_weights = local_batch.to(device).float(), local_labels.to(device).float(), local_weights.to(device).float()
                
                # Update weights
                optimizer.zero_grad()
                outs = model(local_batch).squeeze()
                loss = criterion(outs, local_labels)
                loss = torch.dot(loss, local_weights)
                loss /= local_batch.size()[0]
                loss.backward()
                optimizer.step()
                train_loss_epoch[-1] += loss.item()

                # Calculate accuracy
                predicted = (outs > 0.5).float()
                train_correct += (predicted == local_labels).sum().item()

            train_loss_epoch[-1] /= total_train
            train_acc_epoch.append(train_correct / total_train)
           
            ## GPU Utilization after training
            # print(f"Epoch {epoch + 1}/{max_epochs} GPU Utilization After Training:")
            print_gpu_utilization()

            ## Validation
            model.eval()
            with torch.no_grad():
                for local_batch, local_labels, local_weights in validation_generator:
                    total_val += local_batch.shape[0]

                    # Transfer to GPU and convert to float
                    local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).float()

                    # Evaluate model
                    outs = model(local_batch).squeeze()
                    loss = criterion(outs, local_labels).mean()
                    val_loss_epoch[-1] += loss.item()

                    # Calculate accuracy
                    predicted = (outs > 0.5).float()
                    val_correct += (predicted == local_labels).sum().item()

                val_loss_epoch[-1] /= total_val
                val_acc_epoch.append(val_correct / total_val)

            # Print loss and accuracy in current epoch
            print(f'Fold {fold + 1} - Epoch {epoch + 1}/{max_epochs}\tTrain loss: {train_loss_epoch[-1]:.6f}\tVal loss: {val_loss_epoch[-1]:.6f}\tTrain Acc: {train_acc_epoch[-1]:.6f}\tVal Acc: {val_acc_epoch[-1]:.6f}')

            # Early Stopping
            if val_loss_epoch[-1] < best_val_loss - delta:
                best_val_loss = val_loss_epoch[-1]
                early_stop_count = 0
                torch.save(model.state_dict(), f'/content/drive/MyDrive/Train_New_Dataset_test2/Models/model_fold_{fold + 1}.pt')
            else:
                early_stop_count += 1

            if early_stop_count >= patience:
                print(f"Early stopping on fold {fold + 1} at epoch {epoch + 1}")
                break

            # Update LR and momentum (only if using SGD)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.995
                if 10 <= epoch <= 20:
                    param_group['momentum'] += 0.045

        print(f"Training done for fold {fold + 1}")

        # Ensure the plots directories exist
        os.makedirs('/content/drive/MyDrive/Train_New_Dataset_test2/loss', exist_ok=True)
        os.makedirs('/content/drive/MyDrive/Train_New_Dataset_test2/accuracy', exist_ok=True)

        # Plot and save losses vs epoch
        plt.plot(train_loss_epoch, label='train')
        plt.plot(val_loss_epoch, label='val')
        plt.legend()
        plt.savefig(f'/content/drive/MyDrive/Train_New_Dataset_test2/loss/loss_curves_fold_{fold + 1}.png')
        plt.clf()

        # Plot and save accuracy vs epoch
        plt.plot(train_acc_epoch, label='train')
        plt.plot(val_acc_epoch, label='val')
        plt.legend()
        plt.savefig(f'/content/drive/MyDrive/Train_New_Dataset_test2/accuracy/accuracy_curves_fold_{fold + 1}.png')
        plt.clf()

if __name__ == '__main__':
    folds = 8  # cmd line argument for number of folds
    main(folds)
