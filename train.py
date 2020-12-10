import os
import sys
import time
import torch
from tqdm import tqdm
from collections import OrderedDict
from torchvision import transforms as T
import matplotlib.pyplot as plt

## my library
from model import RCAN
from data_loader import get_loader
from utils import get_gpu_memory, sec2time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
t2img = T.ToPILImage()



""" make result folder """
version = '201209_RCAN'
mode = 'original'

result_dir = f'./results/{version}/{mode}'
weight_dir = f'./weights/{version}'

os.makedirs(result_dir, exist_ok=True)
os.makedirs(weight_dir, exist_ok=True)



""" set training environment """


pfix = OrderedDict()
hist = {'Loss': [], 'Loss S': [], 'Loss T': [], 'Loss D': [], 'Iter': []}

num_epochs = 1000

model = RCAN(scale=2).to(device)

params = list(model.parameters())
optim = torch.optim.Adam(params, lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma= 0.99)
criterion = torch.nn.L1Loss()

height = 128
width = 128
batch_size = 4
augment = True # if True, random crop from image, else, center crop

loader = get_loader(h=height, w=width, augment=augment, device=device, batch_size=batch_size)

save_image_every = 100
save_model_every = 10


""" start training """

start_time = time.time()

print(f'Training Start || Ver: {version} || Mode: {mode}')
step = 0
for epoch in range(num_epochs):
    with tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}', position=0, leave=True) as pbar:
        for lr, hr, _ in pbar:
            
            # prediction
            pred = model(lr)            
            
            # training
            loss = criterion(hr, pred)            
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
            hist['Iter'].append(step)
            hist['Loss'].append(loss.item())
            pbar.set_postfix(pfix)
            
            if step % save_image_every == 0:
            
                z = torch.zeros_like(lr[0])
                xz = torch.cat((lr[0], z), dim=-2)                                
                img = torch.cat((xz, pred[0], hr[0]), dim=-1)                
                img = torch.clamp(img, 0, 1).cpu()
                img = t2img(img)
                img.save(f'{result_dir}/epoch_{epoch+1}_iter_{step:05d}.jpg')
                
                fig = plt.figure(figsize=(20,20))
                ax1 = fig.add_subplot(1,1,1)
                
                ax1.plot(hist['Iter'], hist['Loss'])
                ax1.legend(['Loss'])
                ax1.grid('on')            
                
                fig.savefig(f'{result_dir}/loss.jpg')
                plt.close('all')
            step += 1
        
        if (epoch+1) % save_model_every == 0:
            torch.save(model.state_dict(), f'{weight_dir}/epoch_{epoch+1:04d}_loss_{loss.item():.4f}.pth')



