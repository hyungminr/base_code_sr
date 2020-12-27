import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import time
import torch
from torch.functional import Tensor
from tqdm import tqdm
from collections import OrderedDict
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import shutil

from skimage.metrics import peak_signal_noise_ratio as get_psnr
from skimage.metrics import structural_similarity as get_ssim

# my library
from data_loader import get_loader
from utils import get_gpu_memory, sec2time

import importlib.util
from shutil import copyfile
import torch.fft as fft


if __name__ == '__main__':


    """ config """
    class Config:
        def __init__(self):
            # self.version = '201209_RCAN'
            self.modelpath = './models/model.py'
            self.mode = 'ex7_ffttransformLogscaleNPooling'
            self.height = 128
            self.width = 128
            self.batch_size = 4
            self.num_epochs = 200
            self.scale_factor = 2
            self.augment = True  # if True, random crop from image, else, center crop


    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # from models.model import RCAN





    """ make result folder """
    weight_dir = f'./weights/{config.mode}'
    if os.path.isdir(weight_dir):
        shutil.rmtree(weight_dir)
    os.makedirs(weight_dir, exist_ok=True)

    log_dir = f'./log/{config.mode}'
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    pfix = OrderedDict()


    """ set training environment """

    # model build
    # model = RCAN(scale=config.scale_factor).to(device)
    # make model intance by file path
    spec = importlib.util.spec_from_file_location("model", config.modelpath)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    model = foo.RCAN(scale=config.scale_factor).to(device)

    params = list(model.parameters())
    optim = torch.optim.Adam(params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.99)
    criterion = torch.nn.L1Loss()

    # save train.py & model.py
    copyfile(__file__, f'{log_dir}/{os.path.basename(__file__)}')
    copyfile(config.modelpath, f'{log_dir}/{os.path.basename(config.modelpath)}')



    # dataset
    train_loader, test_loader = get_loader(h=config.height, w=config.width, scale_factor=config.scale_factor,
                                        augment=config.augment, batch_size=config.batch_size)


    """ start training """

    start_time = time.time()

    print(f'Training Start || Mode: {config.mode}')
    step = 0
    for epoch in range(config.num_epochs):

        ## train loop
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', position=0, leave=True) as pbar:
            for lr, hr, _ in pbar:

                lr = lr.to(device)
                hr = hr.to(device)

                hr_fft = torch.fft.fftn(hr, dim =(2,3))
                hr_fft_r = torch.log(torch.abs(hr_fft.real)+1)
                hr_fft_i = torch.log(torch.abs(hr_fft.imag)+1)

                hr_max_p = torch.nn.functional.max_pool2d(hr, kernel_size=3, stride=1)
                hr_min_p = torch.nn.functional.max_pool2d(-hr, kernel_size=3, stride=1)

                
                # prediction
                pred = model(lr)
                pred_fft = torch.fft.fftn(pred, dim =(2,3))
                pred_r = torch.log(torch.abs(pred_fft.real)+1)
                pred_i = torch.log(torch.abs(pred_fft.imag)+1)
                pred_max_p = torch.nn.functional.max_pool2d(pred, kernel_size=3, stride=1)
                pred_min_p = torch.nn.functional.max_pool2d(-pred, kernel_size=3, stride=1)


                # training
                loss = criterion(hr, pred) + 0.1*criterion(hr_fft_r, pred_r) + 0.1*criterion(hr_fft_i, pred_i)\
                    + criterion(hr_max_p, pred_max_p) + criterion(hr_min_p, pred_min_p)


                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()

                # training history
                free_gpu = get_gpu_memory()[0]
                elapsed_time = time.time() - start_time
                elapsed = sec2time(elapsed_time)
                pfix['Step'] = f'{step+1}'
                pfix['Loss'] = f'{loss.item():.4f}'
                pfix['free GPU'] = f'{free_gpu}MiB'
                pfix['Elapsed'] = f'{elapsed}'
                pbar.set_postfix(pfix)

                writer.add_scalar('train/Loss', loss.item()/config.batch_size, step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], step)

                step += 1
                

                
        ## valid loop
        with torch.no_grad():
            print("valid")
            losses = []
            psnr_list = []
            ssim_list = []

            for lr, hr, _ in tqdm(test_loader):
                lr = lr.to(device)
                hr = hr.to(device)
                
                hr_fft = torch.fft.fftn(hr, dim =(2,3))
                hr_fft_r = torch.log(torch.abs(hr_fft.real)+1)
                hr_fft_i = torch.log(torch.abs(hr_fft.imag)+1)

                hr_max_p = torch.nn.functional.max_pool2d(hr, kernel_size=3, stride=1)
                hr_min_p = torch.nn.functional.max_pool2d(-hr, kernel_size=3, stride=1)

                
                # prediction
                pred = model(lr)
                pred_fft = torch.fft.fftn(pred, dim =(2,3))
                pred_r = torch.log(torch.abs(pred_fft.real)+1)
                pred_i = torch.log(torch.abs(pred_fft.imag)+1)
                pred_max_p = torch.nn.functional.max_pool2d(pred, kernel_size=3, stride=1)
                pred_min_p = torch.nn.functional.max_pool2d(-pred, kernel_size=3, stride=1)


                # training
                loss = criterion(hr, pred) + 0.1*criterion(hr_fft_r, pred_r) + 0.1*criterion(hr_fft_i, pred_i)\
                    + criterion(hr_max_p, pred_max_p) + criterion(hr_min_p, pred_min_p)

                # loss
                losses.append(loss.item()/config.batch_size)

                # psnr, ssim
                hr_np = hr.cpu().detach().numpy()
                hr_np = np.transpose(hr_np, (0, 2, 3, 1))
                        
                pred_np = pred.cpu().detach().numpy()
                pred_np = np.transpose(pred_np, (0, 2, 3, 1))

                for i in range(lr.shape[0]):
                    psnr_list.append(get_psnr(hr_np[i], pred_np[i]))
                    ssim_list.append(get_ssim(hr_np[i], pred_np[i], multichannel=True, gaussian_weights=True))

                last_lr = lr
                last_hr = hr
                last_pred = pred

                
            grid_lr = torchvision.utils.make_grid(last_lr)
            grid_hr = torchvision.utils.make_grid(last_pred)
            grid_gt = torchvision.utils.make_grid(last_hr)

            print(f'epoch : {epoch},  loss : {np.mean(losses)}')
            writer.add_scalar('test/Loss', np.mean(losses), step)
            writer.add_scalar('test/PSNR', np.mean(psnr_list), step)
            writer.add_scalar('test/SSIM', np.mean(ssim_list), step)
            writer.add_image('images_lr', grid_lr, step)
            writer.add_image('images_hr', grid_hr, step)
            writer.add_image('images_gt', grid_gt, step)

            print(f'{weight_dir}/epoch_{epoch+1:04d}_loss_{np.mean(losses):.4f}.pth')
            torch.save(model.state_dict(), f'{weight_dir}/epoch_{epoch+1:04d}_loss_{np.mean(losses):.4f}_psnr_{np.mean(psnr_list):.4f}.pth')

