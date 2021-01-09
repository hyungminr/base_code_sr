import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return DRN(args)
    
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

class ResBlock(nn.Module):
    """ Residual Block """
    def __init__(self, kernel=3, padding=1, bias=True, res_scale=1):
        super().__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        self.conv = nn.Sequential(*layers)
        self.res_scale = res_scale
        
    def forward(self, x):
        res = self.conv(x).mul(self.res_scale)
        return x + res
        
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
            
class DRN(nn.Module):
    """  """
    def __init__(self, num_RB=16, num_feats=16, scale=4, kernel=3, padding=1, bias=True):
        super().__init__()
        
        self.up_bicubic = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
        self.sub_mean = MeanShift(mode='sub')
        
        layers = []
        layers += [nn.Conv2d(in_channels= 3, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.head = nn.Sequential(*layers)
        
        
        layers = []
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, stride=2, padding=padding, bias=False)]
        layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True]
        for _ in range(1, int(np.log2(scale))):
            layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, stride=2, padding=padding, bias=False)]
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats*2, kernel_size=kernel, padding=padding, bias=False)]
        self.down_1 = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Conv2d(in_channels=num_feats*2, out_channels=num_feats*2, kernel_size=kernel, stride=2, padding=padding, bias=False)]
        layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True]
        for _ in range(1, int(np.log2(scale))):
            layers += [nn.Conv2d(in_channels=num_feats*2, out_channels=num_feats*2, kernel_size=kernel, stride=2, padding=padding, bias=False)]
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True]
        layers += [nn.Conv2d(in_channels=num_feats*2, out_channels=num_feats*4, kernel_size=kernel, padding=padding, bias=False)]
        self.down_2 = nn.Sequential(*layers)
        
        
        layers = []
        # layers += [ResBlock() for _ in range(num_RB)]
        layers += [ResBlock() for _ in range(16)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.body = nn.Sequential(*layers)
        
        layers = []
        if (scale & (scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats*4, kernel_size=kernel, padding=padding, bias=bias)]
                layers += [nn.PixelShuffle(2)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=3, kernel_size=kernel, padding=padding, bias=bias)]
        self.tail = nn.Sequential(*layers)
        
        self.add_mean = MeanShift(mode='add')
        
    def forward(self, lr):    
    
        lrx4 = self.up_bicubic(lr)
            
        # meanshift (preprocess)
        fx4 = self.sub_mean(lrx4)
        
        # shallow feature
        fx4 = self.head(fx4)
        
        fx2 = self.down_1(fx4)
        fx1 = self.down_2(fx2)
        
        # down feature
        x_down = self.down(x)
        
        
        return out
