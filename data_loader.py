import os
import glob
import numpy as np
import math
import random
import torch
import torchvision.transforms as T
from PIL import Image

class dataset(torch.utils.data.Dataset):
    """ Load HR / LR pair """
    def __init__(self, root_dir, height=128, width=128, augment=False, device='cpu'):
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.augment = augment
        self.files = self.find_files()
        self.device = device
    
    def find_files(self):
        return glob.glob(f'{self.root_dir}/*.png')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
                
        hflip = random.choice([True, False]) if self.augment else False
        vflip = random.choice([True, False]) if self.augment else False
        transform = self.get_transform(hflip, vflip)
        
        h = self.height
        w = self.width
                
        index = self.indexerror(index)
        input_name = self.files[index]
        output_name = input_name.replace('LR_bicubic/X2', 'HR')
        output_name = output_name.replace('x2.png', '.png')
        
        input_image = Image.open(input_name)
        output_image = Image.open(output_name)
        
        crop = self.get_crop_bbox(output_image)
        
        input_image = self.crop_image(input_image, crop, scale_factor=2)
        output_image = self.crop_image(output_image, crop)
        
        input_tensor = transform(input_image)
        output_tensor = transform(output_image)
        
        input_tensor = input_tensor.to(self.device)
        output_tensor = output_tensor.to(self.device)
        
        return input_tensor, output_tensor
        
    def get_transform(self, hflip=False, vflip=False):
        transform = []
        if hflip: transform.append(T.RandomHorizontalFlip(1))
        if vflip: transform.append(T.RandomVerticalFlip(1))
        transform.append(T.ToTensor())
        return T.Compose(transform)
    
    def get_crop_bbox(self, image, scale_factor=1):
        width, height = image.size
        w = self.width // scale_factor
        h = self.height // scale_factor
        if self.augment:
            left = np.random.randint(width - w)
            top = np.random.randint(height - h)
        else:
            left = (width - w) // 2
            top = (height - h) // 2
        right = left + w
        bottom = top + h 
        return [left, top, right, bottom]
        
    def crop_image(self, image, crop_shape, scale_factor=1):
        crop_shape = [i // scale_factor for i in crop_shape]
        return image.crop(crop_shape)

    
    def load_tensor(self, file_name, transform=None):
        image = Image.open(file_name)
        if transform is None:
            transform = get_transform()
        if self.augment:
            image = self.augment(image)
        return transform(image)
    
    def indexerror(self, index):
        index = index if index < len(self.files) else 0
        return index
    

def get_loader(root_dir='./data/DIV2K/DIV2K_train_LR_bicubic/X2/', batch_size=1, num_workers=0, mode='train', h=128, w=128, augment=False, device='cpu'):
        
    shuffle = (mode == 'train')    
    data_loader = torch.utils.data.DataLoader(dataset=dataset(root_dir, h, w, augment, device),
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)   
    return data_loader

