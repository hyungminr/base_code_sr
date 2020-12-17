import os
import glob
import numpy as np
import math
import random
import torch
import torchvision.transforms as T
from PIL import Image

class DatasetDIV2K(torch.utils.data.Dataset):
    """ Load HR / LR pair """
    def __init__(self, root_dir, height=128, width=128, scale_factor=2, augment=False, mode = "train"):
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.augment = augment
        self.files = self.find_files()
        if mode =="train":
            self.files = self.files[:-100]
        else:
            self.files = self.files[-100:]
        
        self.scale_factor = scale_factor
    
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
        
        output_name = self.files[index]
        input_name = output_name.replace('HR', f'LR_bicubic/X{self.scale_factor}')
        input_name = input_name.replace('.png', f'x{self.scale_factor}.png')
               
        input_image = Image.open(input_name)
        output_image = Image.open(output_name)
        
        crop = self.get_crop_bbox(output_image)
        
        input_image = self.crop_image(input_image, crop, scale_factor=self.scale_factor)
        output_image = self.crop_image(output_image, crop)
        
        input_tensor = transform(input_image)
        output_tensor = transform(output_image)
        
        # input_tensor = input_tensor.to(self.device)
        # output_tensor = output_tensor.to(self.device)
        
        return input_tensor, output_tensor, output_name
        
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


    def indexerror(self, index):
        index = index if index < len(self.files) else 0
        return index
    

def get_loader(root_dir='./data/DIV2K/DIV2K_train_HR/', batch_size=1, num_workers=0, h=128, w=128, scale_factor=2, augment=False):

    train_data_loader = torch.utils.data.DataLoader(dataset=DatasetDIV2K(root_dir, h, w, scale_factor, augment, "train"),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,)  
    test_data_loader = torch.utils.data.DataLoader(dataset=DatasetDIV2K(root_dir, h, w, scale_factor, False, "test"),
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,)   
    return train_data_loader, test_data_loader

