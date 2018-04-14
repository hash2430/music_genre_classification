# GCT634 (2018) HW2
#
# Apr-11-2018: initial version
#
# Jongpil Lee
#

from __future__ import print_function
import numpy as np

import torch
from torch.utils.data import DataLoader

import torch.nn as nn


import Baseline.src.NotRunnables.model as model
from Baseline.src.NotRunnables.gtzandata import gtzandata
from Baseline.src.NotRunnables.train_validate import fit, eval
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

############################################################################## constant
melBins = 128
hop = 512
frames = int(29.9*22050.0/hop)
batch_size = 5
learning_rate = 0.01
num_epochs = 50
num_frames = 120 # frames랑 num_frames랑 차이가 뭐지?

# A location where labels and features are located
label_path = './gtzan/'
############################################################################### data preparation
# read train / valid / test lists
y_train_dict = {}
y_valid_dict = {}
y_test_dict = {}
with open(label_path + 'train_filtered.txt') as f:
    train_list = f.read().splitlines()
    for line in train_list:
        y_train_dict[line] = line.split('/')[0]
with open(label_path + 'valid_filtered.txt') as f:
    valid_list = f.read().splitlines()
    for line in valid_list:
        y_valid_dict[line] = line.split('/')[0]
with open(label_path + 'test_filtered.txt') as f:
    test_list = f.read().splitlines()
    for line in test_list:
        y_test_dict[line] = line.split('/')[0]


# labels
genre_set1 = set(y_train_dict.values())
genre_set2 = set(y_valid_dict.values())
genre_set3 = set(y_test_dict.values())
genres = genre_set1 | genre_set2 | genre_set3
genres = list(genres)
print(genres)

# why do you do this? It semms like there's no point doing this.
for iter in range(len(y_train_dict)):
    for iter2 in range(len(genres)):
        if genres[iter2] == y_train_dict[train_list[iter]]:
            y_train_dict[train_list[iter]] = iter2

for iter in range(len(y_valid_dict)):
    for iter2 in range(len(genres)):
        if genres[iter2] == y_valid_dict[valid_list[iter]]:
            y_valid_dict[valid_list[iter]] = iter2

for iter in range(len(y_test_dict)):
    for iter2 in range(len(genres)):
        if genres[iter2] == y_test_dict[test_list[iter]]:
            y_test_dict[test_list[iter]] = iter2


mel_path = './gtzan_mel/'

# load data
x_train = np.zeros((len(train_list),melBins,frames)) #3D ndarray
y_train = np.zeros((len(train_list),)) #1D
for iter in range(len(train_list)):
    x_train[iter] = np.load(mel_path + train_list[iter].replace('.wav','.npy'))
    y_train[iter] = y_train_dict[train_list[iter]]

x_valid = np.zeros((len(valid_list),melBins,frames))
y_valid = np.zeros((len(valid_list),))
for iter in range(len(valid_list)):
    x_valid[iter] = np.load(mel_path + valid_list[iter].replace('.wav','.npy'))
    y_valid[iter] = y_valid_dict[valid_list[iter]]

x_test = np.zeros((len(test_list),melBins,frames))
y_test = np.zeros((len(test_list),))
for iter in range(len(test_list)):
    x_test[iter] = np.load(mel_path + test_list[iter].replace('.wav','.npy'))
    y_test[iter] = y_test_dict[test_list[iter]]

# normalize the mel spectrograms
mean = np.mean(x_train)
std = np.std(x_train)
x_train -= mean
x_train /= std
x_valid -= mean
x_valid /= std
x_test -= mean
x_test /= std

print(x_train.shape,y_train.shape)

train_data = gtzandata(x_train,y_train)
valid_data = gtzandata(x_valid,y_valid)
test_data = gtzandata(x_test,y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last = True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
################################################################################# learning & testing

# load model
model = model.model_1DCNN()

# training & validation
criterion = nn.CrossEntropyLoss()
fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs)

# test
avg_loss, output_all, label_all = eval(model,test_loader,criterion)
#print(len(output_all),output_all[0].shape,avg_loss)

prediction = np.concatenate(output_all)
prediction = prediction.reshape(len(test_list),len(genres))
prediction = prediction.argmax(axis=1)
#print(prediction)

y_label = np.concatenate(label_all)
#print(y_label)

comparison = prediction - y_label
acc = float(len(test_list) - np.count_nonzero(comparison)) / len(test_list)
print('Test Accuracy: {:.4f} \n'. format(acc))

# TODO segmentation eval function average !!!
