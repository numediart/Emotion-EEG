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

from Models import *
from Utils import *
from Utils_Bashivan import *
import time
import warnings

warnings.simplefilter("ignore")

t = time.time()
device = torch.device('cuda:0')

X_images = np.load('.npy') # place here the images representation of EEG
X_array = np.load('.npy') # place here the array representation of EEG features
Label = np.load('.npy') # place here the label for each EEG
Participant = np.load('.npy') # place here the array with each participants

n_epoch = 150

Dataset = CombDataset(label=Label, image=X_image, array=X_array) #creation of
#dataset classs in Pytorch

# electrodes locations in 3D -> 2D projection
locs_3d = np.load('.npy')
locs_2d = []
for e in locs_3d:
    locs_2d.append(azim_proj(e))

for p in range(len(np.unique(Participant))):
    print("Training participant ", p)

    #Splitting in Train and Testing Set
    idx = np.argwhere(Participant == p)[:, 0]
    np.random.shuffle(idx)
    Test = Subset(Dataset, idx)
    idx = np.argwhere(Participant != p)[:, 0]
    np.random.shuffle(idx)
    Train = Subset(Dataset, idx)

    #Train Test Loader Pytorch
    Trainloader = DataLoader(Train, batch_size=128, shuffle=False)
    Testloader = DataLoader(Test, batch_size=128, shuffle=False)

    #Set training parameters
    lr = 1e-3
    wd = 1e-4
    mom= 0.9

    net = MultiModel().to(device)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom,  weight_decay=wd)
    optimizer = optim.Adam(net.parameters(), lr=lr,  weight_decay=wd)

    Res = []

    validation_loss = 0.0
    validation_acc = 0.0

    for epoch in range(n_epoch):
        running_loss = 0.0
        evaluation = []

        #Training
        net.train()
        for i, data, data_test in iter_over(Trainloader, Testloader):
            source_img, source_arr, label = data #signals from training
            target_img, target_arr, _ = data_test #signals with unknwon label

            # Image Representaion
            img = torch.cat([source_img, target_img])
            img = img * mutl_img
            img = img.to(device)

            # Array Representation
            arr = torch.cat([source_arr, target_arr])
            arr = arr.to(device)

            # True Domain
            domain_y = torch.cat([torch.ones(source_img.shape[0]),
                                  torch.zeros(target_img.shape[0])])
            domain_y = domain_y.to(device)

            # True Label
            label = label.to(device)

            # Estimation of both feature vectors + concat
            feat_img = net.FeatCNN(img.to(torch.float32).to(device))
            feat_arr = net.FeatRNN(arr.to(torch.float32).to(device))
            feat = torch.cat((feat_img, feat_arr), axis=1)

            # Estimation of both labels from each models
            label_pred_cnn = net.ClassifierFC_CNN(feat_img[:source_img.shape[0]])
            label_pred_rnn = net.ClassifierFC_CNN(feat_img[:source_img.shape[0]])

            # Computing Loss from labels
            label_loss_cnn = F.cross_entropy(label_pred_cnn, label.long())
            label_loss_rnn = F.cross_entropy(label_pred_rnn, label.long())

            #Combination
            label_loss = label_loss_rnn + label_loss_cnn
            label_pred = label_pred_cnn + label_pred_rnn
            running_loss += label_loss.item()

            # Domain prediction + Loss
            domain_pred = net.Discriminator(feat).squeeze()
            domain_loss = F.binary_cross_entropy_with_logits(domain_pred, domain_y)

            # Loss Backward
            optimizer.zero_grad()
            loss = domain_loss + label_loss
            loss.backward()
            optimizer.step()

            # Prediction and Accuracy
            _, predicted = torch.max(label_pred, 1)
            num_of_true = torch.sum(predicted == label)
            mean = num_of_true/label.shape[0]
            evaluation.append(mean)

        running_loss = running_loss / (i + 1)
        running_acc = sum(evaluation) / len(evaluation)

        evaluation = []
        #Evaluation
        net.eval()
        for j, data in enumerate(Testloader, 0):
            img, arr, label = data

            # Prediction
            pred = net(img.to(torch.float32).to(device), arr.to(torch.float32).to(device))
            loss = F.cross_entropy(pred, label.to(device).long())

            # Loss
            validation_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(pred.cpu().data, 1)
            evaluation.append((predicted == label).tolist())

        validation_loss = validation_loss /(j+1)
        evaluation = [item for sublist in evaluation for item in sublist]
        validation_acc = sum(evaluation) / len(evaluation)

        print('[%d, %3d]\tloss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
                      (epoch + 1, n_epoch, running_loss, running_acc, validation_loss, validation_acc))

        Res.append((epoch + 1, n_epoch, running_loss, running_acc, validation_loss, validation_acc))
    path =  'res/sub'+str(p)+'/'
    np.save(path+'rec_'+str(len(glob.glob(path+'*.npy')))+'_lr_'+str(lr)+'_wd_'+str(wd)+'_mom_'+str(mom), Res)
    print('End after_'+str(np.round(time.time() - t, 3))+'\n')
    t = time.time()
