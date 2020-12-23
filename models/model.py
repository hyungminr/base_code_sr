import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=1.0, rgb_mean=(0.4488,0.4371,0.4040), rgb_std=(1.0,1.0,1.0), mode='sub'):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        sign = -1 if mode == 'sub' else 1
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

class RCAB(nn.Module):
    """ Residual Channel Attention Block """
    def __init__(self, kernel=3, padding=1, bias=True):
        super().__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        self.conv_in = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Conv2d(in_channels=64, out_channels= 4, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels= 4, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.Sigmoid()]
        self.channel_att = nn.Sequential(*layers)
        
    def forward(self, x):
        x_fea = self.conv_in(x)
        x_cha = self.channel_att(x_fea)
        x_att = x_fea * x_cha
        return x + x_att

class RG(nn.Module):
    """ Residual Group """
    def __init__(self, num_RCAB=16, kernel=3, padding=1, bias=True):
        super().__init__()
        
        layers = []
        layers += [RCAB() for _ in range(num_RCAB)]
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        self.rcab = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.rcab(x)
    
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
    
class RCAN(nn.Module):
    """  """
    def __init__(self, num_RG=10, scale=2, kernel=3, padding=1, bias=True):
        super().__init__()
        
        self.sub_mean = MeanShift(mode='sub')
        
        layers = []
        layers += [nn.Conv2d(in_channels= 3, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        self.head = nn.Sequential(*layers)
        
        layers = []
        layers += [RG() for _ in range(num_RG)]
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        self.body = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        self.body_last = nn.Sequential(*layers)
        
        layers = []
        if (scale & (scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                layers += [nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=kernel, padding=padding, bias=bias)]
                layers += [nn.PixelShuffle(2)]
        layers += [nn.Conv2d(in_channels=64, out_channels=3, kernel_size=kernel, padding=padding, bias=bias)]
        self.tail = nn.Sequential(*layers)
        
        self.add_mean = MeanShift(mode='add')
        
    def forward(self, img):    
    
        # meanshift (preprocess)
        x = self.sub_mean(img)
        
        # shallow feature
        x_shallow = self.head(x)
        
        # deep feature
        x_deep = self.body(x_shallow)
        
        # shallow + deep
        x_feature = x_shallow + x_deep
        
        # upscale
        x_up = self.tail(x_feature)
        
        # meanshift (postprocess)
        out = self.add_mean(x_up)
        
        return out
