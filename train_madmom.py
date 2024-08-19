import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import glob
import os
import sys
from torch.utils import data
from utils import Dataset
import matplotlib.pyplot as plt
from class_madmom import MadmomToTorchNN

def load_madmom_model(model_path):
    with open(model_path, 'rb') as f:
        madmom_model = pickle.load(f, encoding='latin1')
    return madmom_model


def main(fold):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    params = {'batch_size': 256, 'shuffle': True, 'num_workers': 6}
    max_epochs = 50

    datadir = 'Small_Dataset/data_pt_test'
 
    with open('songlist.txt', 'r') as file:
        songlist = file.read().splitlines()
    labels = np.load('labels_master.npy', allow_pickle=True).item()
    weights = np.load('weights_master.npy', allow_pickle=True).item()
  
    madmom_model_path = 'onsets_cnn.pkl'
    madmom_model = load_madmom_model(madmom_model_path)
    model = MadmomToTorchNN(madmom_model).double().to(device)

    criterion = torch.nn.BCELoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.45)
  
    fold = int(sys.argv[1])
    partition = {'all': [], 'train': [], 'validation': []}
  
    dataset_folder = 'Small_Dataset/splits'
    file_name = '8-fold_cv_random_%d.fold' % fold
    file_path = os.path.join(dataset_folder, file_name)

    with open(file_path, 'r') as file:
        val_split = file.readlines()

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

    with open('labels.txt', 'w') as file:
        file.write(str(labels))

    with open('partition.txt', 'w') as file:
        file.write(str(partition))

    n_ones = 0.
    for idi in partition['train']:
        if labels[idi] == 1.:
            n_ones += 1
    print('Fraction of positive examples: %f' % (n_ones / len(partition['train'])))

    training_set = Dataset(partition['train'], labels, weights)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'], labels, weights)
    validation_generator = data.DataLoader(validation_set, **params)
    print("Generator done")
  
    print("Training...")
        
    train_loss_epoch = []
    val_loss_epoch = []
    for epoch in range(max_epochs):
        train_loss_epoch.append(0)
        val_loss_epoch.append(0)

        n_train = 0
        for local_batch, local_labels, local_weights in training_generator:
            n_train += local_batch.shape[0]
            local_batch, local_labels, local_weights = local_batch.to(device), local_labels.to(device), local_weights.to(device)

            optimizer.zero_grad()
            outs = model(local_batch).squeeze()
            loss = criterion(outs, local_labels)
            loss = torch.dot(loss, local_weights)
            loss /= local_batch.size()[0]
            loss.backward()
            optimizer.step()
            train_loss_epoch[-1] += loss.item()
        train_loss_epoch[-1] /= n_train
        
        n_val = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels, local_weights in validation_generator:
                n_val += local_batch.shape[0]
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                outs = model(local_batch).squeeze()
                loss = criterion(outs, local_labels).mean()
                val_loss_epoch[-1] += loss.item()
        val_loss_epoch[-1] /= n_val

        print('Epoch no: %d/%d\tTrain loss: %f\tVal loss: %f' % (epoch, max_epochs, train_loss_epoch[-1], val_loss_epoch[-1]))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.995
            if 10 <= epoch <= 20:
                param_group['momentum'] += 0.045

    print("Training done")
    os.makedirs('./plots', exist_ok=True)
    plt.plot(train_loss_epoch, label='train')
    plt.plot(val_loss_epoch, label='val')
    plt.legend()
    plt.savefig('./plots/loss_curves_%d.png' % fold)
    plt.clf()
    torch.save(model.state_dict(), 'saved_model_fold_%d.pt' % fold)

if __name__ == '__main__':
    fold = int(sys.argv[1])
    main(fold)
