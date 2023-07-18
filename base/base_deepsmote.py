import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np



class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding,bias=False)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x
    
class UpSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, stride=2, tan_activation=False, batchnorm=True):
        super().__init__()
        self.tan_activation = tan_activation
        self.batchnorm = batchnorm

        self.convT = nn.ConvTranspose2d(in_channels,out_channels,kernel,stride,padding=1)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if tan_activation:
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU(True)

    def forward(self,x):
        
        x = self.convT(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.tan_activation:
            x = self.act(x)
        
        return x


    
class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        layers = nn.ModuleList()
        layers.append(DownSampleConv(self.n_channel, self.dim_h, batchnorm=False))

        values=range(4)
        for index, item in enumerate(values[1:], 1):
            layers.append(DownSampleConv(self.dim_h * (2 ** values[index-1]),self.dim_h * (2 ** item), batchnorm=True))
        
        for _ in range(3):
            layers.append(DownSampleConv(self.dim_h * (2 ** values[-1]),self.dim_h * (2 ** values[-1]), batchnorm=True))


        self.layers = layers
        self.fc = nn.Linear(self.dim_h * (2 ** values[-1]), self.n_z)

    
    def forward(self, input):
       
        output = input
        for layer in self.layers:
            output = layer(output)
        
        output = output.squeeze()

        output = self.fc(output)
    

        return output
    

  
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']



        
        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            
            #3d and 32 by 32
            #nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 1, 0, bias=False),
            
            nn.BatchNorm2d(self.dim_h * 8), # 40 X 8 = 320
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True) )#,
            #nn.Conv2d(self.dim_h * 8, 1, 2, 1, 0, bias=False))
            #nn.Conv2d(self.dim_h * 8, 1, 4, 1, 0, bias=False))
        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)
        
class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']


        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU())
        
        layers = nn.ModuleList()

        for _ in range(2):
            layers.append(UpSampleConv(self.dim_h * 8,self.dim_h * 8))

        layers.append(UpSampleConv(self.dim_h * 8,self.dim_h * 4))
        layers.append(UpSampleConv(self.dim_h * 4,self.dim_h * 2))
        layers.append(UpSampleConv(self.dim_h * 2,3,stride=2,tan_activation=True))

        self.layers = layers

    def forward(self, input):
       
        output = input

        output = self.fc(output)
        output = output.view(-1, self.dim_h * 8, 7, 7)
        for layer in self.layers:
            output = layer(output)
    

        return output  
       

        

