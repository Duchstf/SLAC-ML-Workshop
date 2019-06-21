"""
Author: Duc Hoang.
Date: June, 20, 2019

Simple CNN model for bunch2bunch project
"""

from __future__ import print_function

#Neuralnet stuff
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch import Tensor

#Data Processing
import matplotlib
import numpy as np

#Timing
import time

#Initialize random seed 
SEED=123
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

###---DEFINE THE MODEL---###
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # feature extractor CNN
        self._feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(6,32,3,padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32,32,3,padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2,2),

            torch.nn.Conv2d(32,64,3,padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64,64,3,padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2,2),

            torch.nn.Conv2d(64,128,3,padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128,128,3,padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2,2),

            torch.nn.Conv2d(128,256,3,padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256,256,3,padding=1),
            torch.nn.LeakyReLU())
        # Regressor
        self._regressor = torch.nn.Sequential(
            torch.nn.Linear(7680, 16000),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16000, 16000),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16000, 12000))

        def weights_init(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)

        self._feature_extractor.apply(weights_init)

    def forward(self, x):
        # extract features
        features = self._feature_extractor(x)
        # flatten the 3d tensor (2d space x channels = features)
        features = features.view(-1, np.prod(features.size()[1:]))

        # regress and return
        return self._regressor(features)

###---LOSS AND OPTIMIZER---###
def createLossAndOptimizer(net, learning_rate=0.001):

    #Loss function
    loss = torch.nn.SmoothL1Loss()

    #Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    return(loss, optimizer)

import time

###---TRAIN FUNCTION---###
def trainNet(net, batch_size, n_epochs, learning_rate, X, y):

    #NOTE: X,y are numpy arrays


    # Create dataset from several tensors with matching first dimension
    # Samples will be drawn from the first dimension (rows)
    dataset = TensorDataset(Tensor(X), Tensor(y))

    #Split the data into train and validation set (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    #Initialize loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 27)

    #Number of batches
    n_batches = len(train_loader)

    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    #Time for printing
    training_start_time = time.time()

    #Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):

            #Get inputs
            inputs, labels = data

            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            #Switch to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            #Set the parameter gradients to zero
            optimizer.zero_grad()

            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            #Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()

            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
             #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            #Switch to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

if __name__ == "__main__":
    # Read in data 
    X = np.load('IN_CNN.npy')
    y = np.load('OUT_CNN.npy')

    #Run on GPU in available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Run all the stuffs together
    CNN = CNN()
    CNN.to(device)

    trainNet(CNN, batch_size=30, n_epochs=20, learning_rate=0.001, X=X, y=y)
                                                                                