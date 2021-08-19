import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()  # Inherited all the methods of the class nn.Module
        """
        
        DEFINE YOUR NETWORK HERE
        
        """
        self.num_classes = num_classes
        # CNN with 16 filters and 8x8x1 conv with stride 2 and padding 0
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=2, bias=True)

        # Leaky ReLU with 0.1 leak for negative input
        self.l_relu = nn.LeakyReLU(0.1)

        # CNN with 32 filters and 4x4x16 conv with stride 2 and padding 0
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=True)

        # Maxpool Layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # CNN with K filters and 6x6x32 conv with stride 2 and padding 0
        self.naive = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=6, stride=1, bias=True)

        # Softmax
        self.soft = nn.Softmax2d()

    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """
        x = self.conv1(x)
        x = self.l_relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.l_relu(x)
        x = self.maxpool(x)

        # change this obviously!
        # x = torch.flatten(x, 1)
        # out = nn.Linear(x,  self.num_classe)
        # print(x.shape)

        out = self.naive(x)

        # For h part of the question. Not applying the softmax layer as it is already included in nn.CrossEntropy function.

        return out
