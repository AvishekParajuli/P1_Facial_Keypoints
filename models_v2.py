## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # Net architecture IN->Conv2d(3x3,16)->Maxpool->Conv2d(3x3,32)->Maxpool->conv2d(3x3,64)->conv2d(3x3,128)
        # -->MaxPool->[[conv2d(3x3,256)]]->flatten->dense(-1,..)->dropout(0.4)->dense2(-1,138)
        # we will use 3x3 conv throughout and use padding to maintain same size from conv; 
        # pooling will be used exclusively to decrease size
        
        # 1 input image channel (grayscale),
        #input image size = 224x224 (rescaled and cropped+ normalized)
        self.conv1 = nn.Conv2d(1, 16, 3, padding =1)
        #output size = (W-F+2P)/S +1 = (224-3+2)/1 +1 = 224; same size as previous
        # then maxpool => (112x112, 16)
        #2 conv
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        #output size = (W-F+2P)/S +1 = (112-3+2)/1 +1 = 112
        #maxpool=>(56x56, 32)
        ## Define the rest of the layers:
        self.conv3 = nn.Conv2d(32, 64, 3,padding=1)
        #maxpool=> (28x28, 64)
        self.conv4 = nn.Conv2d(64, 64, 3,padding=1)
        #=> maxpool=> (14x14, 64)
        # flatten => 14x14x64
        self.conv_drop = nn.Dropout(p=0.1)
        
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        #dropout with prob =0.5
        self.fc1_drop = nn.Dropout(p=0.4)
        #linear dense layers
        self.fc1 = nn.Linear(14*14*64, 500)
        self.fc2 = nn.Linear(500, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # Net architecture IN->Conv2d(3x3,16)->Maxpool->Conv2d(3x3,32)->Maxpool->conv2d(3x3,64)->conv2d(3x3,128)
        # -->MaxPool->[opt[conv2d(3x3,256)]]->flatten->dense(-1,..)->dropout(0.4)->dense2(-1,138)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_drop(x)
        
        #
        x = F.relu(self.conv3(x))
        x= self.pool(x)
        x = self.conv_drop(x)
        x = F.relu(self.conv4(x))
        x= self.pool(x)
        #flatten (14x14, 64)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
