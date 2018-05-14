from __future__ import print_function, division
import torch
from math import *
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F

train_data = []
train_labels = []
X = np.load("train.npy")
Y = np.load("test.npy")
X = np.concatenate((X,np.ones((X.__len__(),1))),axis=1)
Y = np.concatenate((Y,np.ones((Y.__len__(),1))),axis=1)
train_data = X[:,1:]
train_labels = X[:,0]
test_data = Y[:,1:]
test_labels = Y[:,0]

mean=np.mean(train_data,0)
std=np.std(train_data,0)
std[std[:]<0.00001]=1
train_data=(train_data-mean)/std
test_data=(test_data-mean)/std

h = 1
pd = 3
nf = int(train_data.shape[1])
nc = int(np.max(train_labels))
train_labels = np.array(train_labels - np.min(train_labels),dtype=int)
test_labels = np.array(test_labels - np.min(test_labels),dtype=int)

lT = 1e-2
lW = 1e-2
lV = 1e-2
lZ = 1e-3
bs = 100
sig = 4

class Bonsai(nn.Module):
    def __init__(self,h,pd,nf,nc,lT,lW,lV,lZ,sig):
        super(Bonsai, self).__init__()
        self.h = h
        self.int_n = int(pow(2,self.h) - 1)
        self.tot_n = int(pow(2,(self.h + 1)) - 1)
        self.pd = pd
        self.nf = nf
        self.nc = nc

        self.lT = lT
        self.lW = lW
        self.lV = lV
        self.lZ = lZ

        self.sig = sig
        self.sigI = 1
        self.Z = nn.Parameter(torch.FloatTensor(torch.rand(self.pd, self.nf))-0.5)

        if(self.int_n > 0):
            self.T = nn.Parameter(torch.FloatTensor(torch.rand(self.int_n, self.pd, 1))-0.5)

        self.V = nn.Parameter(torch.FloatTensor(torch.rand(self.tot_n, self.pd, self.nc)) - 0.5)
        self.W = nn.Parameter(torch.FloatTensor(torch.rand(self.tot_n, self.pd, self.nc)) - 0.5)

    def forward(self, x):
        bs = x.size(0)
        pp = torch.matmul(self.Z,x.view(bs,self.nf,1))/self.pd
        pp = pp.view(bs,self.pd)

        I = Variable(torch.FloatTensor(torch.ones(bs, self.tot_n)))
        score = Variable(torch.FloatTensor(torch.zeros(bs, self.nc)))

        if(self.int_n > 0):
            for i in range(1,self.tot_n):
                j = int(floor((i + 1) / 2) - 1)
                I[:, i] = 0.5 * I[:, j] * (1 + pow(-1, (i + 1) - 2 * (j + 1)) * F.tanh(self.sigI * torch.t(torch.matmul(pp, self.T[j]))))

        for i in range(self.tot_n):
            score = score + torch.t(torch.t(torch.matmul(pp, self.W[i]) * F.tanh(self.sig * torch.matmul(pp, self.V[i]))) * I[:, i])
        return score

    def multi_class_loss(self, outputs, labels):
        reg_loss = 0.5 * (self.lW * torch.norm(self.W) + self.lV * torch.norm(self.V) + self.lZ * torch.norm(self.Z))
        if(self.int_n > 0):
            reg_loss += 0.5 * (self.lT * torch.norm(self.T))
        class_function = nn.CrossEntropyLoss()
        class_loss = class_function(outputs,labels)
        total_loss = reg_loss + class_loss
        return total_loss

bonsai = Bonsai(h,pd,nf,nc,lT,lW,lV,lZ,sig)
loss_function = lambda x,y: bonsai.multi_class_loss(x,y)
optimizer = optim.Adam(bonsai.parameters(),lr=0.005)

total_epochs = 10
num_iters = int(train_data.shape[0]/bs)
total_batches = num_iters * total_epochs

for i in range(total_epochs):
    acc = 0
    for j in range(num_iters):
        optimizer.zero_grad()
        inputs, labels = torch.FloatTensor(train_data[j * bs:(j + 1) * bs, :]).view(bs, nf), torch.LongTensor(train_labels[j * bs:(j + 1) * bs]).view(bs)
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = bonsai(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

    acc = 0
    sigI_old = bonsai.sigI
    bonsai.sigI = 1e9

    for j in range(int(test_data.__len__() / bs)):
        inputs, labels = torch.FloatTensor(test_data[j * bs:(j + 1) * bs, :]).view(bs, nf), torch.LongTensor(test_labels[j * bs:(j + 1) * bs]).view(bs)
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = bonsai(inputs)
        _, predictions = torch.max(outputs, 1)
        acc += ((predictions == labels).sum()).data.numpy()[0]

    bonsai.sigI = sigI_old
    print("Test Accuracy after Iteration",i+1,"=",end=' ')
    print(float(100 * acc)/Y.__len__(),"%")

    if(bonsai.sigI < 1e9):
        bonsai.sigI *= 2

'''
t = bonsai.T.view(bonsai.int_n * bonsai.pd).data.numpy()
w = bonsai.W.view(bonsai.tot_n * bonsai.pd * bonsai.nc).data.numpy()
v = bonsai.V.view(bonsai.tot_n * bonsai.pd * bonsai.nc).data.numpy()
z = bonsai.Z.data.view(bonsai.pd * bonsai.nf).numpy()

values = np.concatenate((t,w,v,z,mean,std,test_data[0]),axis=0)
print("[",end='')
for i in range(values.shape[0]):
    print("%0.4f" % values[i],end=',')
    if(i % 20 == 0):
        print()
print("]")
'''