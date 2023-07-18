#%%
import os
import tarfile
from PIL import Image
from torch.utils import data
import numpy as np


processed_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"./SORTIT/images")



def default_loader(path):
    return Image.open(path)


def make_dataset():
    
    file = tarfile.open('./sortit.tar.gz')
    file.extractall('.')
    file.close()
    
    images = []
    processed_train_img_path = processed_train_path
    processed_train_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"./SORTIT/masks")
    for img_name in os.listdir(processed_train_img_path):
        item = (os.path.join(processed_train_img_path, img_name),
                os.path.join(processed_train_mask_path , img_name))
        images.append(item)
    
    return images


class resortit(data.Dataset):
    def __init__(self,  simul_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset()
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.loader = default_loader
        self.simul_transform = simul_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = self.loader(img_path)
        mask = np.array(self.loader(mask_path))
        # if cfg.DATA.NUM_CLASSES == 1:
        #     mask[mask>0] = 1   ##########Only Binary Segmentation#####
        mask = Image.fromarray(mask)
        if self.simul_transform is not None:
            img, mask = self.simul_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
