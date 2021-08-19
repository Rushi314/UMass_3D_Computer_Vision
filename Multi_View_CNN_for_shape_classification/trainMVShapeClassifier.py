from __future__ import division
import numpy as np
import torch.utils.data as utils_data
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import CNN
from data_utils import *


def trainMVShapeClassifier(dataset_path, cuda=False, verbose=False):
    """
    this function trains a multi-view convnet for 3D shape classification
    YOU HAVE TO MODIFY THIS FUNCTION!

    dataset_path is the name of the folder that contains PNG images of
    rendered 3D shapes. The folder should have the following structure:
      => category_1 [folder]
          => shape1_id [folder]
              => shape1_id_001.png [grayscale image]
              => shape1_id_002.png [grayscale image]
                 ...
          => shape2_id [folder]
              => shape2_id_001.png [grayscale image]
              => shape2_id_002.png [grayscale image]
                 ...
          => ...
      => category_2 [folder]
         ...
    the function should return a trained convnet
    """
    # Read all PNG images and their category information, and save them in numpy matrix.
    # (could be commented out after first run
    # save_data(dataset_path)

    # Load saved numpy matrix
    data, info = load_data()
    print(info)
    # Get train/validation data, X: images, Y: labels
    X, Xval = data['X'][data['train']], data['X'][data['val']]
    Y, Yval = data['Y'][data['train']], data['Y'][data['val']]
    print("X Shape = ", X.shape)
    print("Y Shape = ", Y.shape)
    X = Variable(torch.from_numpy(X))
    Y = Variable(torch.from_numpy(Y)).type(torch.LongTensor)
    Xval = Variable(torch.from_numpy(Xval))
    Yval = Variable(torch.from_numpy(Yval)).type(torch.LongTensor)
    if cuda:
        X, Y, Xval, Yval = X.cuda(), Y.cuda(), Xval.cuda(), Yval.cuda()

    if len(X) == 0:
        print("Error Loading Data!")
        return

    # An interactive plot showing how the loss function and classification error behave on the training and
    # validation split during learning.
    fig, axes = plt.subplots(ncols=3, nrows=1)
    axes[0].set_title('Loss')
    axes[1].set_title('Top1 Error')
    axes[2].set_title('Top5 Error')
    plt.tight_layout()  # Adjusts the padding between and around the subplots
    plt.ion()  # Turn the interactive mode on
    plt.show()

    # Model
    model = CNN(num_classes=len(info['category_names']))
    if cuda:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    count = 0
    learningRate = 0.001
    numEpochs = 20
    weightDecay = 0.0001
    momentum = 0.9
    batch_size = info['num_views']
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightDecay)

    training_samples = utils_data.TensorDataset(X, Y)
    data_loader = utils_data.DataLoader(training_samples, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Training Starting..")
    for epoch in range(numEpochs):
        ''' ======================================
                      TRAIN SECTION
        ====================================== '''
        train_loss = 0
        train_accuracy = []
        for i, batch in enumerate(data_loader):

            # load images and labels in batch
            images, labels = batch[0], batch[1]

            # forward pass
            optimizer.zero_grad()  # Will Zero the gradients for every loop.i.e will not allow the gradients to accumulate.
            outputs = model(images).squeeze()

            # loss
            loss = criterion(outputs, labels)

            # back propagation
            loss.backward()
            optimizer.step()

            # report loss/accuracy scores
            if cuda:
                train_loss += loss.data.cpu().numpy()
                train_accuracy.append(predictions(labels.data.cpu().numpy(), outputs.data.cpu().numpy()))
            else:
                train_loss += loss.data.numpy()
                train_accuracy.append(predictions(labels.data.numpy(), outputs.data.numpy()))

            if verbose:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f' % (
                    epoch + 1, numEpochs, i, len(X) // batch_size, train_loss / (i + 1)))

        train_accuracy = np.mean(train_accuracy, axis=0)

        ''' ======================================
                   VALIDATION SECTION
        ====================================== '''
        output_val = model(Xval).squeeze()
        if cuda:
            val_loss = criterion(output_val, Yval).data.cpu().numpy()
            val1, val5 = predictions(Yval.data.cpu().numpy(), output_val.data.cpu().numpy())
        else:
            val_loss = criterion(output_val, Yval).data.numpy()
            val1, val5 = predictions(Yval.data.numpy(), output_val.data.numpy())

        # Draw an interactive plot showing how the loss function and classification error
        # behave on the training and validation split during learning.
        if epoch != 0:
            # pause required to update the graph
            axes[0].plot([epoch - 1, epoch], [prevtra, train_loss / (i + 1)], marker='o', color="blue", label="train")
            plt.pause(0.0001)
            axes[0].plot([epoch - 1, epoch], [prevval, val_loss], marker='o', color="red", label="val")
            plt.pause(0.0001)
            axes[1].plot([epoch - 1, epoch], [prevtop1_train, 1 - train_accuracy[0]], marker='o', color="blue",
                         label="train")
            plt.pause(0.0001)
            axes[1].plot([epoch - 1, epoch], [prevtop1_val, 1 - val1], marker='o', color="red", label="val")
            plt.pause(0.0001)
            axes[2].plot([epoch - 1, epoch], [prevtop5_train, 1 - train_accuracy[1]], marker='o', color="blue",
                         label="train")
            plt.pause(0.0001)
            axes[2].plot([epoch - 1, epoch], [prevtop5_val, 1 - val5], marker='o', color="red", label="val")
            plt.pause(0.0001)
            if epoch == 1:
                axes[0].legend(loc='upper right')
                axes[1].legend(loc='upper right')
                axes[2].legend(loc='upper right')

        # calculate accuracy scores for plotting
        prevval, prevtra = val_loss, train_loss / (i + 1)
        prevtop1_val, prevtop5_val = 1 - val1, 1 - val5
        prevtop1_train, prevtop5_train = 1 - train_accuracy[0], 1 - train_accuracy[1]

        # report scores per epoch
        print('Epoch [%d/%d], Loss: %.4f, Top 1 Error: %.3f, Top 5 Error: %.3f' % (
            epoch + 1, numEpochs, train_loss / (i + 1), 1 - train_accuracy[0], 1 - train_accuracy[1]))

        # save trained models
        save_checkpoint(model, epoch)

        # save loss/accuracy figures
        plt.savefig("errorplot.png")

    return model, info
