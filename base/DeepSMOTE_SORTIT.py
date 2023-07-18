
#%%
import imageio.v2  as imageio
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms as standard_transforms
import transforms as own_transforms
from resortit import resortit
from base import Encoder,Decoder
from PIL import Image,ImageOps
import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported

attr_map = dict(item.strip().split('=', 1) for item in sys.argv[1:])
run = neptune.init_run(
    project="stiver/DeepSmote-Seg",  # replace with your own (see instructions below)
    api_token=attr_map['token'],
)



##############################################################################
"""args for AE"""

args = {}
args['dim_h'] = 64         # factor controlling size of hidden layers
args['n_channel'] = 3    # number of channels in the input data 

args['n_z'] = 300     # number of dimensions in latent space. 

args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
args['epochs'] = 200       # how many epochs to run for
args['batch_size'] = 16   # batch size for SGD
       # save weights at each epoch of training if True


args['dataset'] = 'mnist'  #'fmnist' # specify which dataset to use


##############################################################################

def check_image(image,mask):
    #inputs, labels = data
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    #restore_transform(inputs[0])
    plt.imshow(image, cmap='gray')
    plt.subplot(122)
    plt.imshow(mask, cmap='gray')
    plt.show()


##############################################################################
"""set models, loss functions"""
# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False



def sample():
    images= []
    tc = np.random.choice(range(5)[1:],1)
    index=np.argwhere(np.any(dec_y == tc , axis=(1, 2)))
    index=index.squeeze()


    result_list = [(img_transform(Image.open(f_images[i])),mask_transform(Image.fromarray(dec_y[i])))for i in index] 
    
    for image,mask in result_list:
        image[:,mask != tc.item()] = 0
        images.append(image)
    print("samples of class {} ".format(tc))
    
    return np.stack(images)
  




mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

img_transform = standard_transforms.Compose([
    transforms.Resize((224, 224)),                  
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
mask_transform = standard_transforms.Compose([
    transforms.Resize((224, 224)), 
    own_transforms.MaskToTensor(),
    own_transforms.ChangeLabel(255, 4)
])

restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        transforms.Resize((800, 800)),
        standard_transforms.ToPILImage(),
    ])



train_set = resortit( transform=img_transform,
                           target_transform=mask_transform)


images_list  = [(f_image,imageio.imread(f_mask)) for f_image,f_mask in train_set.imgs]
print("whole data")
f_images, dec_y = zip(*images_list)
f_images = list(f_images)
dec_y = list(dec_y)


train_loader = torch.utils.data.DataLoader(train_set, 
        batch_size=args['batch_size'],shuffle=True,num_workers=0)

encoder = Encoder(args)
decoder = Decoder(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
decoder = decoder.to(device)
encoder = encoder.to(device)

train_on_gpu = torch.cuda.is_available()

#decoder loss function
criterion = nn.MSELoss()
criterion = criterion.to(device)
    


best_loss = np.inf



enc_optim = torch.optim.Adam(encoder.parameters(), lr = args['lr'])
dec_optim = torch.optim.Adam(decoder.parameters(), lr = args['lr'])

autoencoder = nn.Sequential(*[encoder, decoder])

npt_logger = NeptuneLogger(run,model=autoencoder, log_model_diagram=True, log_parameters=True)
run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(args)

for epoch in range(args['epochs']):
    train_loss = 0.0
    tmse_loss = 0.0
    tdiscr_loss = 0.0
    # train for one epoch -- set nets to train mode
    encoder.train()
    decoder.train()

    for images,labs in train_loader:
    
        # zero gradients for each batch
        encoder.zero_grad()
        decoder.zero_grad()
        #print(images)
        images, labs = images.to(device), labs.to(device)
        #print('images ',images.size()) 
        labsn = labs.detach().cpu().numpy()
        #print('labsn ',labsn.shape, labsn)
    
        # run images
        z_hat = encoder(images)
    
        x_hat = decoder(z_hat) #decoder outputs tanh
        #print('xhat ', x_hat.size())
        #print(x_hat)
        #it's feed with general dataset
        mse = criterion(x_hat,images)

                
        resx = []
        resy = []

        samples = sample()
        print("sale")
        xlen = len(samples)
        nsamp = min(xlen, 100)
        ind = np.random.choice(list(range(len(samples))),nsamp,replace=False)
        xclass = samples[ind]
        # yclass = ybeg[ind]
    
        xclen = len(xclass)
        xcminus = np.arange(1,xclen)
        xcplus = np.append(xcminus,0)
        xcnew = (xclass[xcplus,:])
        # #xcnew = np.squeeze(xcnew)
        #xcnew = xcnew.reshape(xcnew.shape[1],xcnew.shape[2],xcnew.shape[3],xcnew.shape[4])
        # #print('xcnew ',xcnew.shape)
    
        xcnew = torch.Tensor(xcnew)
        xcnew = xcnew.to(device)
    
        #encode xclass to feature space
        xclass = torch.Tensor(xclass)
        xclass = xclass.to(device)
        xclass = encoder(xclass)
        #print('xclass ',xclass.shape) 
    
        xclass = xclass.detach().cpu().numpy()
    
        xc_enc = (xclass[xcplus,:])
        #xc_enc = np.squeeze(xc_enc)
        #print('xc enc ',xc_enc.shape)
    
        xc_enc = torch.Tensor(xc_enc)
        xc_enc = xc_enc.to(device)
        
        ximg = decoder(xc_enc)
        #it's feed with imbalanced class of dataset
        mse2 = criterion(ximg,xcnew)
    
        comb_loss = mse2 + mse
        comb_loss.backward()
    
        enc_optim.step()
        dec_optim.step()
    
        train_loss += comb_loss.item()
        tmse_loss += mse.item()
        tdiscr_loss += mse2.item()
            
    # print avg training statistics
    run[npt_logger.base_namespace]["train_loss/epoch/"].append(train_loss/len(train_loader))
    run[npt_logger.base_namespace]["tmse_loss/epoch/"].append(tmse_loss/len(train_loader))
    run[npt_logger.base_namespace]["tdiscr_loss/epoch/"].append(tmse_loss/len(train_loader))  
    train_loss = train_loss/len(train_loader)



    #store the best encoder and decoder models
    #here, /crs5 is a reference to 5 way cross validation, but is not
    #necessary for illustration purposes
    if train_loss < best_loss:
        path_enc = 'bst_enc.pth'
        path_dec = 'bst_dec.pth'
        # run["best_encoder"].upload(encoder)
        # run["best_decoder"].upload(decoder)
        torch.save(encoder.state_dict(), 'bst_enc.pth')
        torch.save(decoder.state_dict(), 'bst_dec.pth')

        best_loss = train_loss

    # run["final_encoder"].upload(encoder)
    # run["final_decoder"].upload(decoder)

    torch.save(encoder.state_dict(), 'fnl_enc.pth')
    torch.save(decoder.state_dict(), 'fnl_dec.pth')