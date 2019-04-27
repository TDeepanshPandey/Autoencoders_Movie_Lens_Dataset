# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:16:33 2019

@author: tdpco
"""

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import time

# Importing the Dataset
movie = pd.read_csv('ml-1m/movies.dat',
                    sep='::',
                    header=None,
                    engine='python',
                    encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat',
                    sep='::',
                    header=None,
                    engine='python',
                    encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',
                      sep='::',
                      header=None,
                      engine='python',
                      encoding='latin-1')

# Dividing into test and training set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = max(max(training_set[:, 0]),
               max(test_set[:, 0]))
nb_movies = max(max(training_set[:, 1]),
                max(test_set[:, 1]))


def convert(data):
    '''
    Converting the data into an array with user in line and movie in column
    '''
    new_data = []
    for id_users in range(1, nb_users+1):
        # Creating a list of list for one user and movie rating
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_rating = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_rating
        new_data.append(list(ratings))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


class SAE(nn.Module):
    '''
    Creating the architecture of the neural network
    '''
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

start_time = time.time()
# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0)+1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step()
    print('Epoch : '+str(epoch)+' Train Loss - '+str(train_loss/s))
end_time = time.time()
print('Autoencoder Model Training Time : '+str(round((end_time-start_time)/60, 0))+' Minutes.')

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0)+1e-10)
        test_loss += np.sqrt(loss.item()*mean_corrector)
        s += 1.
print(' Train Loss - '+str(train_loss/s))
