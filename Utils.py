'''
Created by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
victor.delvigne@umons.ac.be
Source: SEEN SOON
Copyright (C) 2019 - UMons
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
'''


import numpy as np
import math as m

import torch.nn as nn
import torch

import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.interpolate import griddata
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from sklearn.preprocessing import scale
from scipy.signal import resample
from torch.utils.data import DataLoader, random_split, Subset

'''
def array_to_epochs(data, channels, sampling_frequency, montage='standard_1020', channel_type=['eeg']):
    channel_type = channel_type * len(channels)
    info = mne.create_info(ch_names=channels, sfreq=sampling_frequency, ch_types=channel_type,
                           montage=mne.channels.make_standard_montage(montage), verbose=50)

    event_id, tmin, tmax = 1, -1., data.shape[1] / sampling_frequency + 0.5
    baseline = (None, 0)
    events = np.array([[100, 0, event_id]])
    epochs = mne.EpochsArray(data.reshape(1, data.shape[0], data.shape[1]), info=info, events=events,
                             event_id={'arbitrary': 1}, verbose=50)
    return epochs


def compute_psd(epoch, fmin=-1., fmax=60.):
    psds, freqs = psd_multitaper(epoch, fmin=fmin, fmax=fmax, n_jobs=10, verbose=50)
    return resample(psds, num=1500, axis=2)[0, :]

'''

def image_generation(feature_matrix, electrodes_loc, n_gridpoints):
    n_electrodes = electrodes_loc.shape[0]  # number of electrodes
    n_bands = feature_matrix.shape[1] // n_electrodes  # number of frequency bands considered in the feature matrix
    n_samples = feature_matrix.shape[0]  # number of samples to consider in the feature matrix.

    # Checking the dimension of the feature matrix
    if feature_matrix.shape[1] % n_electrodes != 0:
        print('The combination feature matrix - electrodes locations is not working.')
    assert feature_matrix.shape[1] % n_electrodes == 0
    new_feat = []

    # Reshape a novel feature matrix with a list of array with shape [n_samples x n_electrodes] for each frequency band
    for bands in range(n_bands):
        new_feat.append(feature_matrix[:, bands * n_electrodes: (bands + 1) * n_electrodes])

    # Creation of a meshgrid data interpolation
    #   Creation of an empty grid
    grid_x, grid_y = np.mgrid[
                     np.min(electrodes_loc[:, 0]): np.max(electrodes_loc[:, 0]): n_gridpoints * 1j,  # along x_axis
                     np.min(electrodes_loc[:, 1]): np.max(electrodes_loc[:, 1]): n_gridpoints * 1j  # along y_axis
                     ]

    interpolation_img = []
    #   Interpolation
    #       Creation of the empty interpolated feature matrix
    for bands in range(n_bands):
        interpolation_img.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))
    #   Interpolation between the points
    print('Signals interpolations.')
    for sample in range(n_samples):
        for bands in range(n_bands):
            interpolation_img[bands][sample, :, :] = griddata(electrodes_loc, new_feat[bands][sample, :], (grid_x, grid_y), method='cubic', fill_value=np.nan)
    #   Normalization - replacing the nan values by interpolation
    for bands in range(n_bands):
        interpolation_img[bands][~np.isnan(interpolation_img[bands])] = scale(interpolation_img[bands][~np.isnan(interpolation_img[bands])])
        interpolation_img[bands] = np.nan_to_num(interpolation_img[bands])
    return np.swapaxes(np.asarray(interpolation_img), 0, 1) # swap axes to have [samples, colors, W, H]


class EEGImagesDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""

    def __init__(self, label, image):
        self.label = label
        self.Images = image

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.Images[idx]
        label = self.label[idx]
        sample = (image, label)

        return sample

class CombDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""

    def __init__(self, label, image, array):
        self.label = label
        self.array = array
        self.Images = image

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.Images[idx]
        label = self.label[idx]
        array = self.array[idx]
        sample = (image, array, label)

        return sample


def Test_Model(net, Testloader, criterion, is_cuda=True):
    running_loss = 0.0
    evaluation = []
    criterion_ae = nn.MSELoss()
    # y_pred = []
    # y_true = []
    for i, data in enumerate(Testloader, 0):
        input_img, labels = data
        input_img = input_img.to(torch.float32)
        if is_cuda:
            input_img = input_img.cuda()
        outputs = net(input_img)
        _, predicted = torch.max(outputs.cpu().data, 1)
        evaluation.append((predicted == labels).tolist())
        # y_pred.extend(predicted)
        # y_true.extend(labels)
        loss = criterion(outputs, labels.long().cuda()) # + criterion_ae(out_ae, input_img)
        running_loss += loss.item()
    running_loss = running_loss / (i + 1)
    evaluation = [item for sublist in evaluation for item in sublist]
    running_acc = sum(evaluation) / len(evaluation)
    # plt.hist(y_pred, bins=4, rwidth=0.5, alpha=0.5)
    # plt.hist(y_true, bins=4, rwidth=0.5, alpha=0.5)
    # plt.show()
    return running_loss, running_acc


def TrainTest_Model(model, trainloader, testloader, n_epoch=30, opti='SGD', learning_rate=0.0001, is_cuda=True,
                    print_epoch=5, verbose=False):
    if is_cuda:
        net = model().cuda()
    else:
        net = model()

    criterion = nn.CrossEntropyLoss()
    criterion_AE = nn.MSELoss()

    if opti == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    elif opti == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    else:
        print("Optimizer: " + optim + " not implemented.")

    for epoch in range(n_epoch):
        running_loss = 0.0
        evaluation = []
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(torch.float32).cuda())
            _, predicted = torch.max(outputs.cpu().data, 1)
            evaluation.append((predicted == labels).tolist())
            loss = criterion(outputs, labels.long().cuda()) # + criterion_AE(out_ae, inputs.to(torch.float32).cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss = running_loss / (i + 1)
        evaluation = [item for sublist in evaluation for item in sublist]
        running_acc = sum(evaluation) / len(evaluation)
        net.eval()
        validation_loss, validation_acc = Test_Model(net, testloader, criterion, True)

        if epoch % print_epoch == (print_epoch - 1):
            print('[%d, %3d]\tloss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
                  (epoch + 1, n_epoch, running_loss, running_acc, validation_loss, validation_acc))
    if verbose:
        print('Finished Training \n loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
              (running_loss, running_acc, validation_loss, validation_acc))

    return (running_loss, running_acc, validation_loss, validation_acc)

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def iter_over(train_loader, test_loader):
    iter_test_loader = iter(test_loader)

    for i, data_train in enumerate(train_loader, 0):
        try:
            data_test = next(iter_test_loader)
        except StopIteration:
            iter_test_loader = iter(test_loader)
            data_test = next(iter_test_loader)
        yield i, data_train, data_test
