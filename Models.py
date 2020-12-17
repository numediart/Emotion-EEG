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

import torch
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.optim as optim

from torch.autograd import Function

from Utils import *

device = torch.device('cuda:0')

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)



class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class FreqCNN(nn.Module):

    def __init__(self, input_image=torch.zeros(1, 3, 32, 32), kernel=(3, 3), stride=1, padding=1, max_kernel=(2, 2),
                 n_classes=4):
        super(FreqCNN, self).__init__()

        self.ClassifierCNN = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            Flatten(),

            nn.Linear(128,64)
        )
        '''
        self.ClassifierFC = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Dropout(0.5),
            #nn.Softmax(),
        )

        self.Discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        '''

    def forward(self, x):
        x = self.ClassifierCNN(x)
        x = x.view(x.shape[0], -1)
        #x = self.ClassifierFC(x.view(x.shape[0], -1))
        return x


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):

    def forward(self, input):
        return input.view(input.shape[0], -1, 1, 1)


class NotSquare(nn.Module):

    def __init__(self, input_image=torch.zeros(1, 3, 32, 32), kernel=(3, 3), stride=1, padding=1, max_kernel=(2, 2),
                 n_classes=4):
        super(NotSquare, self).__init__()

        self.ClassifierCNN = nn.Sequential(
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 32, kernel_size=3, padding=(0,1)),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),


            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),


            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

        )

        self.ClassifierFC = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Linear(128, 4),
            # nn.Dropout(0.5),
            nn.Softmax(),
        )

        self.Discriminator = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.ClassifierCNN(x)
        # print(x.shape)
        x = self.ClassifierFC(x.view(x.shape[0], -1))
        return x

class RRNN(nn.Module):

    def __init__(self, ChanDict):
        super(RRNN, self).__init__()

        self.dict = ChanDict

        self.hidden_size = 32
        self.num_layers = 2
        self.input_size = 5

        self.bidirectional = False

        self.rnn1 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc1 = nn.Sequential(
            nn.Linear(6*32, 32),
            nn.ReLU()
            )

        self.rnn2 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc2 = nn.Sequential(
            nn.Linear(6*32, 32),
            nn.ReLU()
            )

        self.rnn3 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc3 = nn.Sequential(
            nn.Linear(5*32, 32),
            nn.ReLU()
            )

        self.rnn4 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc4 = nn.Sequential(
            nn.Linear(5*32, 32),
            nn.ReLU()
            )

        self.rnn5 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc5 = nn.Sequential(
            nn.Linear(9*32, 32),
            nn.ReLU()
            )

        self.rnn6 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc6 = nn.Sequential(
            nn.Linear(9*32, 32),
            nn.ReLU()
            )

        self.rnn7 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc7 = nn.Sequential(
            nn.Linear(5*32, 32),
            nn.ReLU()
            )

        self.rnn8 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc8 = nn.Sequential(
            nn.Linear(5*32, 32),
            nn.ReLU()
            )

        self.rnn9 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc9 = nn.Sequential(
            nn.Linear(2*32, 32),
            nn.ReLU()
            )

        self.rnn10 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.fc10 = nn.Sequential(
            nn.Linear(2*32, 32),
            nn.ReLU()
            )

        self.rnn_l = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)
        self.rnn_r = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = False, bidirectional = self.bidirectional)

        self.fc_l = nn.Sequential(
            nn.Linear(5*32, 96),
            nn.ReLU(),
            )

        self.fc_r = nn.Sequential(
            nn.Linear(5*32, 96),
            nn.ReLU(),
            )

        #self.lin1 = nn.Linear((27, 64), (16, 6))

        self.ClassifierCNN = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=2),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(96,4),
            #nn.Softmax(),
            )

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)# 2 for bidirection

        k = list(self.dict.keys())

        out1 , _ = self.rnn1(x[:, self.dict[k[0]], :].permute(1,0,2), h0)
        out1 = out1.permute(1, 0, 2)
        out1 = self.fc1(out1.reshape(self.batch_size, -1))
        #out1 = out1.mean(axis=0)

        out2 , _ = self.rnn2(x[:, self.dict[k[1]], :].permute(1,0,2), h0)
        out2 = out2.permute(1, 0, 2)
        out2 = self.fc2(out2.reshape(self.batch_size, -1))
        #out2 = out2.mean(axis=0)

        out3 , _ = self.rnn3(x[:, self.dict[k[2]], :].permute(1,0,2), h0)
        out3 = out3.permute(1, 0, 2)
        out3 = self.fc3(out3.reshape(self.batch_size, -1))
        #out3 = out3.mean(axis=0)

        out4 , _ = self.rnn4(x[:, self.dict[k[3]], :].permute(1,0,2), h0)
        out4 = out4.permute(1, 0, 2)
        out4 = self.fc4(out4.reshape(self.batch_size, -1))
        #out4 = out4.mean(axis=0)

        out5 , _ = self.rnn5(x[:, self.dict[k[4]], :].permute(1,0,2), h0)
        out5 = out5.permute(1, 0, 2)
        out5 = self.fc5(out5.reshape(self.batch_size, -1))
        #out5 = out5.mean(axis=0)

        out6 , _ = self.rnn6(x[:, self.dict[k[5]], :].permute(1,0,2), h0)
        out6 = out6.permute(1, 0, 2)
        out6 = self.fc6(out6.reshape(self.batch_size, -1))
        #out6 = out6.mean(axis=0)

        out7 , _ = self.rnn7(x[:, self.dict[k[6]], :].permute(1,0,2), h0)
        out7 = out7.permute(1, 0, 2)
        out7 = self.fc7(out7.reshape(self.batch_size, -1))
        #out7 = out7.mean(axis=0)

        out8 , _ = self.rnn8(x[:, self.dict[k[7]], :].permute(1,0,2), h0)
        out8 = out8.permute(1, 0, 2)
        out8 = self.fc8(out8.reshape(self.batch_size, -1))
        #out8 = out8.mean(axis=0)

        out9 , _ = self.rnn9(x[:, self.dict[k[8]], :].permute(1,0,2), h0)
        out9 = out9.permute(1, 0, 2)
        out9 = self.fc9(out9.reshape(self.batch_size, -1))
        #out9 = out9.mean(axis=0)

        out10 , _ = self.rnn10(x[:, self.dict[k[9]], :].permute(1,0,2), h0)
        out10 = out10.permute(1, 0, 2)
        out10 = self.fc10(out10.reshape(self.batch_size, -1))
        #out10 = out10.mean(axis=0)

        x_l = torch.stack([out1, out5, out3, out7, out9])

        x_r = torch.stack([out2, out6, out4, out8, out10])

        x_l, _ = self.rnn_l(x_l, h0)
        x_r, _ = self.rnn_r(x_r, h0)

        x_l = x_l.permute(1, 0, 2)
        x_r = x_r.permute(1, 0, 2)

        x_l = self.fc_l(x_l.reshape(self.batch_size, -1)).view(self.batch_size, 16, 6)
        x_r = self.fc_r(x_r.reshape(self.batch_size, -1)).view(self.batch_size, 16, 6)

        x = x_l + x_r

        #x = self.ClassifierCNN(x)

        x = self.fc(x.view(self.batch_size, -1))
        return x


class BiHDM(nn.Module):

    def __init__(self):
        super(BiHDM, self).__init__()

        self.hidden_size = 32
        self.num_layers = 2
        self.input_size = 5

        self.batch_first = False
        self.bidirectional = False

        self.RNN_VL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_VR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_V = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_HL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_HR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_H = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.fc_v = nn.Sequential(
            nn.Linear(864, 96),
            nn.ReLU(),
            )

        self.fc_h = nn.Sequential(
            nn.Linear(864, 96),
            nn.ReLU(),
            )

        self.fc_c = nn.Sequential(
            nn.Linear(96, 4),
            #nn.LogSoftmax(),
            )


    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)# 2 for bidirection

        VL_id = [0, 3, 5, 6, 7, 8, 14, 15, 16, 17, 23, 24, 25, 26, 32, 33, 34, 35, 41, 42, 43, 44, 50, 51, 52, 58, 57]
        VR_id = [2, 4, 13, 12, 11, 10, 22, 21, 20, 19, 31, 30, 29, 28, 40, 39, 38, 37, 49, 48, 47, 46, 56, 55, 54, 60, 61]

        HL_id = [0, 5, 14, 23, 32, 41, 50, 57, 3, 6, 15, 24, 33, 42, 51, 58, 7, 16, 25, 34, 43, 52, 8, 17, 26, 35, 44]
        HR_id = [2, 13, 22, 31, 40, 49, 56, 61, 4, 12, 21, 30, 39, 48, 55, 60, 11, 20, 29, 38, 47, 54, 10, 19, 28, 37, 46]
        eps = 1e-12
        x_vl, _ = self.RNN_VL(x[:, VL_id].permute(1, 0, 2), h0)
        x_vr, _ = self.RNN_VL(x[:, VR_id].permute(1, 0, 2), h0)

        x_v, _ = self.RNN_V(x_vr - x_vl, h0)

        x_hl, _ = self.RNN_HL(x[:, HL_id].permute(1, 0, 2), h0)
        x_hr, _ = self.RNN_HL(x[:, HR_id].permute(1, 0, 2), h0)

        x_h, _ = self.RNN_V(x_hr - x_hl, h0)

        x_v = self.fc_v(x_v.permute(1, 0, 2).reshape(x.shape[0], -1)).view(x.shape[0],-1)

        x_h = self.fc_h(x_h.permute(1, 0, 2).reshape(x.shape[0], -1)).view(x.shape[0], -1)

        x = x_v + x_h

        #x = self.fc_c(x.view(x.shape[0],-1))

        #x = torch.cat([out1- out2, out3- out4, out5- out6, out7- out8, out9 -out10]).permute(1,0,2)

        #x = self.ClassifierCNN(x)

        #x = self.fc(x.reshape(x.shape[0], -1))
        return x




class FrontalRnn(nn.Module):

    def __init__(self):
        super(FrontalRnn, self).__init__()

        self.FeatRNN = nn.Sequential(
            RegionRNN(32, 1, 5),
            )

        #self.ClassifierFC = nn.Sequential(
        #    nn.Linear(64, 16),
        #    nn.Dropout(0.25),
        #    nn.ReLU(),
        #    nn.Linear(16, 4),
        #    )

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]

        x = self.FeatRNN(x)
        #x = self.ClassifierFC(x)
        #x = self.fc(x.reshape(self.batch_size, -1))
        return x



class RegionRNN(nn.Module):

    def __init__(self, h_size, n_layer, in_size, b_first = False, bidir = False):
        super(RegionRNN, self).__init__()

        self.hidden_size = h_size
        self.num_layers = n_layer
        self.input_size = 5

        self.dict = {'Fr1': np.array([0, 3, 8, 7, 6, 5]), 'Fr2': np.array([ 2,  4, 10, 11, 12, 13]), 'Tp1': np.array([14, 23, 32, 41, 50]),
        'Tp2': np.array([22, 31, 40, 49, 56]), 'Cn1': np.array([15, 16, 17, 26, 25, 24, 33, 34, 35]), 'Cn2': np.array([21, 20, 19, 28, 29, 30, 39, 38, 37]),
        'Pr1': np.array([42, 43, 44, 52, 51]), 'Pr2': np.array([48, 47, 46, 54, 55]), 'Oc1': np.array([58, 57]), 'Oc2': np.array([60, 61])}

        self.batch_first = b_first
        self.bidirectional = bidir

        self.RNN_fL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_fR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_f = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_tL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_tR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_t = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_pL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_pR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_p = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_oL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_oR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_o = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.fc_f = nn.Sequential(
            nn.Linear(6*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.fc_t = nn.Sequential(
            nn.Linear(5*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.fc_p = nn.Sequential(
            nn.Linear(5*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.fc_o = nn.Sequential(
            nn.Linear(2*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.b_n1 = nn.BatchNorm2d(5)
        self.b_n2 = nn.BatchNorm1d(64)

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]

        x = self.b_n1(x.permute(0,2,1).view(x.shape[0], 5, 1, -1 ))[:,:,0].permute(0,2,1)

        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

        k = list(self.dict.keys())

        fr_l = x[:, self.dict[k[0]]].permute(1, 0, 2)
        fr_r = x[:, self.dict[k[1]]].permute(1, 0, 2)

        tp_l = x[:, self.dict[k[2]]].permute(1, 0, 2)
        tp_r = x[:, self.dict[k[2]]].permute(1, 0, 2)

        p_l = x[:, self.dict[k[6]]].permute(1, 0, 2)
        p_r = x[:, self.dict[k[7]]].permute(1, 0, 2)

        o_l = x[:, self.dict[k[8]]].permute(1, 0, 2)
        o_r = x[:, self.dict[k[9]]].permute(1, 0, 2)

        x_fl, _ = self.RNN_fL(fr_l, h0)
        x_fr, _ = self.RNN_fR(fr_r, h0)

        x_tl, _ = self.RNN_tL(tp_l, h0)
        x_tr, _ = self.RNN_tR(tp_r, h0)

        x_pl, _ = self.RNN_tL(p_l, h0)
        x_pr, _ = self.RNN_tR(p_r, h0)

        x_ol, _ = self.RNN_oL(o_l, h0)
        x_or, _ = self.RNN_oR(o_r, h0)

        x_f = x_fr - x_fl
        x_t = x_tr - x_tl
        x_p = x_pr - x_pl
        x_o = x_or - x_ol

        x_f, _  = self.RNN_f(x_f, h0)
        x_t, _  = self.RNN_f(x_t, h0)
        x_p, _  = self.RNN_p(x_p, h0)
        x_o, _  = self.RNN_o(x_o, h0)

        x_f = x_f.permute(1, 0, 2)
        x_t = x_t.permute(1, 0, 2)
        x_p = x_p.permute(1, 0, 2)
        x_o = x_o.permute(1, 0, 2)

        x = torch.cat((self.fc_f(x_f.reshape(self.batch_size, -1)), self.fc_t(x_t.reshape(self.batch_size, -1)),
            self.fc_p(x_p.reshape(self.batch_size, -1)), self.fc_o(x_o.reshape(self.batch_size, -1))), dim=1)

        x = self.b_n2(x)
        #x = self.fc_f(x_f.reshape(self.batch_size, -1))  +  self.fc_t(x_t.reshape(self.batch_size, -1)) + self.fc_p(x_p.reshape(self.batch_size, -1)) + self.fc_o(x_o.reshape(self.batch_size, -1))
        #x = torch.cat((self.fc_f(x_f.reshape(self.batch_size, -1)), self.fc_t(x_t.reshape(self.batch_size, -1))), dim=1)
        x = x.reshape(self.batch_size, -1)
        #x = self.fc(x.reshape(self.batch_size, -1))
        return x


class MultiModel(nn.Module):

    def __init__(self):
        super(MultiModel, self).__init__()

        self.FeatRNN = nn.Sequential(
            RegionRNN(32, 1, 5),
            )

        self.FeatCNN = FreqCNN()

        #self.b_n = nn.BatchNorm1d(128)

        self.ClassifierFC = nn.Sequential(
            nn.Linear(128, 16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(16, 8),
            )

        self.Discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            )


    def forward(self, image, array):
        # Set initial states
        self.batch_size = image.shape[0]

        array = self.FeatRNN(array)
        image = self.FeatCNN(image)

        #x = self.b_n(torch.cat((array, image), axis=1))
        x = torch.cat((image,array), axis=1)
        x = self.ClassifierFC(x)

        return x
