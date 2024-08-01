import sys
import glob
import torch
from torch.utils import data
from utils import onsetCNN, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to repeat positive samples to improve data balance
def balance_data(ids, labels):
    ids2add = []
    for idi in ids:
        print(idi)
        if labels[idi] == 1:
            ids2add.append(idi)
            ids2add.append(idi)
            ids2add.append(idi)
    ids.extend(ids2add)
    return ids

def main(fold):
    # Use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Parameters for data loader
    params = {'batch_size': 256, 'shuffle': True, 'num_workers': 6}
    max_epochs = 50

    # Data
    datadir = '/content/drive/MyDrive/Dataset_cnn/data_pt_test'
    with open('/content/drive/MyDrive/Dataset_cnn/songlist.txt', 'r') as file:
        songlist = file.read().splitlines()
    labels = np.load('/content/drive/MyDrive/Dataset_cnn/labels_master.npy', allow_pickle=True).item()
    weights = np.load('/content/drive/MyDrive/Dataset_cnn/weights_master.npy', allow_pickle=True).item()

    # Model
    model = onsetCNN().double().to(device)
    criterion = torch.nn.BCELoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.45)
    # optimizer=torch.optim.Adam(model.parameters(), lr=0.05)

    # Cross-validation loop
    partition = {'all': [], 'train': [], 'validation': []}
    val_split = np.loadtxt(f'/content/drive/MyDrive/Dataset_cnn/splits/8-fold_cv_random_{fold}.fold', dtype='str')
    for song in songlist:
        ids = glob.glob(os.path.join(datadir, song, '*.pt'))
        if song in val_split:
            partition['validation'].extend(ids)
        else:
            partition['train'].extend(ids)

    # Balance data
    partition['train'] = balance_data(partition['train'], labels)

    # Print data balance percentage
    n_ones = 0.
    for idi in partition['train']:
        if labels[idi] == 1.:
            n_ones += 1
    print('Fraction of positive examples: %f' % (n_ones / len(partition['train'])))

    # Generators
    training_set = Dataset(partition['train'], labels, weights)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'], labels, weights)
    validation_generator = data.DataLoader(validation_set, **params)

    # Training epochs loop
    train_loss_epoch = []
    val_loss_epoch = []
    best_val_loss = float('inf')

    for epoch in range(max_epochs):
        train_loss_epoch += [0]
        val_loss_epoch += [0]

        ## Training
        n_train = 0
        for local_batch, local_labels, local_weights in training_generator:
            n_train += local_batch.shape[0]

            # Transfer to GPU
            local_batch, local_labels, local_weights = local_batch.to(device), local_labels.to(device), local_weights.to(device)

            # Update weights
            optimizer.zero_grad()
            outs = model(local_batch).squeeze()
            loss = criterion(outs, local_labels)
            loss = torch.dot(loss, local_weights)
            loss /= local_batch.size()[0]
            loss.backward()
            optimizer.step()
            train_loss_epoch[-1] += loss.item()
        train_loss_epoch[-1] /= n_train

        ## Validation
        n_val = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels, local_weights in validation_generator:
                n_val += local_batch.shape[0]

                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Evaluate model
                outs = model(local_batch).squeeze()
                loss = criterion(outs, local_labels).mean()
                val_loss_epoch[-1] += loss.item()
        val_loss_epoch[-1] /= n_val

        # Print loss in current epoch
        print('Epoch no: %d/%d\tTrain loss: %f\tVal loss: %f' % (epoch, max_epochs, train_loss_epoch[-1], val_loss_epoch[-1]))

        # Check if this is the best model so far and save it with the epoch number
        if val_loss_epoch[-1] < best_val_loss:
            best_val_loss = val_loss_epoch[-1]
            best_model_path = f'best_model_{fold}_epoch_{epoch}.pt'
            torch.save(model.state_dict(), best_model_path)

        # Update LR and momentum (only if using SGD)
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.995
            if 10 <= epoch <= 20:
                param_group['momentum'] += 0.045

    # Plot losses vs epoch
    os.makedirs('./plots', exist_ok=True)
    plt.plot(train_loss_epoch, label='train')
    plt.plot(val_loss_epoch, label='val')
    plt.legend()
    plt.savefig(f'./plots/loss_curves_{fold}')
    plt.clf()
    torch.save(model.state_dict(), f'saved_model_{fold}.pt')

if __name__ == '__main__':
    fold = int(sys.argv[1])  # cmd line argument
    main(fold)
