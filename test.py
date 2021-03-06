import os
import sys
import time
import torch
from tqdm import tqdm
from collections import OrderedDict
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np

# from skimage.measure import compare_ssim as get_ssim
# from skimage.measure import compare_psnr as get_psnr
from skimage.metrics import peak_signal_noise_ratio as get_psnr
from skimage.metrics import structural_similarity as get_ssim

## my library
from model import RCAN
from data_loader import get_loader
from utils import get_gpu_memory, sec2time

save_results = True # save result images
device = 'cuda' if torch.cuda.is_available() else 'cpu'
t2img = T.ToPILImage()

model_path = './weights/RCAN.pth'
data_path = './data/DIV2K/DIV2K_valid_LR_bicubic/X2/'

if save_results:
    version = '201209_RCAN_test'
    result_dir = f'./results/{version}/'
    os.makedirs(result_dir, exist_ok=True)

""" set training environment """

scale_factor = 2
model = RCAN(scale=scale_factor).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

height = 512
width = 512
batch_size = 1
augment = False # if True, random crop from image, else, center crop
mode = 'test'

loader = get_loader(root_dir=data_path, h=height, w=width, scale_factor=scale_factor, augment=augment, device=device, batch_size=batch_size, mode=mode)


""" start evaluating """

start_time = time.time()

hist = {'psnr': [], 'ssim': []}
pfix = OrderedDict()

print(f'Testing Start || Ver: {version} || Mode: {mode}')

with tqdm(loader, desc=f'Evaluating', position=0, leave=True) as pbar:
    for lr, hr, image_name in pbar:
        
        # prediction
        with torch.no_grad():
            pred = model(lr)            
        
        # test
        hr_np = hr[0].cpu().detach().numpy()
        hr_np = np.transpose(hr_np, (1, 2, 0))
        
        pred_np = pred[0].cpu().detach().numpy()
        pred_np = np.transpose(pred_np, (1, 2, 0))
        
        psnr = get_psnr(hr_np, pred_np)
        ssim = get_ssim(hr_np, pred_np, multichannel=True, gaussian_weights=True)
                
        hist['psnr'].append(psnr)
        hist['ssim'].append(ssim)   
        
        psnr_mean = np.array(hist['psnr']).mean()
        ssim_mean = np.array(hist['ssim']).mean()
        
        pfix['psnr'] = f'{psnr:.4f}'
        pfix['ssim'] = f'{ssim:.4f}'
        pfix['psnr_mean'] = f'{psnr_mean:.4f}'
        pfix['ssim_mean'] = f'{ssim_mean:.4f}'
        
        pbar.set_postfix(pfix)
        print(image_name)
        if save_results:
            file_name = image_name[0].split('/')[-1]
            z = torch.zeros_like(lr[0])
            xz = torch.cat((lr[0], z), dim=-2)
            img = torch.cat((xz, pred[0], hr[0]), dim=-1)           
            img = torch.clamp(img, 0, 1).cpu()
            img = t2img(img)
            img.save(f'{result_dir}/{file_name}')


