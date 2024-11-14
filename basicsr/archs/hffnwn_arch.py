
from tkinter import N
from tkinter.font import nametofont
from turtle import forward
from grpc import Channel
from numpy import pad
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
# from basicsr.archs.rcan_arch import *
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import Upsample, make_layer
from einops import rearrange
from timm.models.layers.helpers import to_2tuple
from basicsr.archs.arch_util import default_init_weights

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x
    
def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def default_conv(wn,in_channels, out_channels, kernel_size, bias=True):
    return wn(nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias))


class CCAwn(nn.Module):
    def __init__(self, wn,channel, reduction=16):
        super(CCAwn, self).__init__()

        self.contrast = stdv_channels
        self.conv_du = nn.Sequential(
            wn(nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True)),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )
    def forward(self, x):
        res=x
        y = self.contrast(x)
        y = self.conv_du(y)
        return res*y
       
class CCALwn(nn.Module):
    def __init__(self,wn, channel, reduction=16):
        super(CCALwn, self).__init__()     
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            wn(nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x*y
    
class AVGCwn(nn.Module):
    def __init__(self,wn, channel, reduction=16):
        super(AVGCwn, self).__init__()

        # self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            wn(nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y
      
class CCALayerwn(nn.Module):
    def __init__(self, wn, channel, reduction=16):
        super(CCALayerwn, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            wn(nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x*y
    
    # 和omni中esa的区别 omni中f=nfeat 没有gelu
class ESAwn(nn.Module):
    def __init__(self,wn, num_feat=48, conv=nn.Conv2d):
        super(ESAwn, self).__init__()
        f = num_feat // 4
        self.conv1 = wn(nn.Conv2d(num_feat, f, 1))
        self.conv2 = wn(conv(f, f, 3, 2, 0))
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv3 = wn(conv(f, f, kernel_size=3, padding=1))
        self.conv4 = wn(nn.Conv2d(f, f, 1))
        self.conv5 = wn(nn.Conv2d(f, num_feat, 1))
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1 = (self.conv1(input))
        c2 = self.conv2(c1)
        pool = self.maxPooling(c2)
        c3 = self.GELU(self.conv3(pool))
        # c3 = self.GELU(self.conv3(v_range))
        # c3 = self.conv3_(c3)
        up = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        c4 = self.conv4(c1)
        c5 = self.conv5((up + c4))
        m = self.sigmoid(c5)

        return input * m
## Pixel Attention Layer
class PALayerwn(nn.Module):
    def __init__(self,wn, channel, reduction=16, bias=True):
        super(PALayerwn, self).__init__()
        self.pa = nn.Sequential(
            wn(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=bias)),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y
    
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

## Features Attention include Channel attention and Pixel Attention 006
class FABlockwn(nn.Module):
    def __init__(self, wn,dim, reduction=16,kernel_size=3 ):
        super(FABlockwn, self).__init__()
        # self.conv1 = wn(conv(dim, dim, kernel_size, bias=True))
        self.conv3 = wn(conv(dim, dim//3, kernel_size=3,  bias=True))
        self.conv5 = wn(conv(dim, dim//3, kernel_size=5, stride=1 ,bias=True))
        self.conv7 = wn(conv(dim, dim//3, kernel_size=7, bias=True))
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = wn(conv(dim, dim, kernel_size, bias=True))
        self.calayer = CCALwn(wn=wn,channel=dim,reduction=reduction)
        self.palayer = PALayerwn(wn,dim,reduction=reduction)

    def forward(self, x):
        res1=self.conv3(x)
        res2=self.conv5(x)
        res3=self.conv7(x)
        res=self.act1(torch.cat([res1,res2,res3],dim=1))
        # res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res = res + x
        return res  
    
    
class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class FTB(nn.Module):
    def __init__(self,n_feat):
        super(FTB,self).__init__()
        self.conv1=nn.Conv2d(n_feat,n_feat,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp=ConvMlp(n_feat)
        self.prelu=nn.PReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        res=x
        x=self.prelu(self.conv1(x))
        x=self.avg_pool(x)
        x=self.mlp(x)
        x=self.sigmoid(x)
        return x*res
    
class LFFB(nn.Module):
    def __init__(self,wn,n_feat):
        super(LFFB,self).__init__()
        # 06
        # self.hffb1=HFFB(wn,n_feat)
        # self.hffb2=HFFB(wn,n_feat)
        # self.hffb3=HFFB(wn,n_feat)
        # self.hffb4=HFFB(wn,n_feat)
        # self.hffb5=HFFB(wn,n_feat)
        # 08
        self.hffb1=HFFBlock(wn,n_feat)
        self.hffb2=HFFBlock(wn,n_feat)
        self.hffb3=HFFBlock(wn,n_feat)
        # self.hffb4=HFFBlock(wn,n_feat)
        # self.hffb5=HFFBlock(wn,n_feat)
        # self.conv1=wn(nn.Conv2d(n_feat*5,n_feat,1))
        # self.conv2=nn.Conv2d(n_feat,n_feat,1)
        # self.conv3=nn.Conv2d(n_feat,n_feat,1)
        # self.conv4=nn.Conv2d(n_feat,n_feat,kernel_size=3,stride=1,padding=1)
        
        self.conv1=wn(nn.Conv2d(n_feat*3,n_feat,1))
        self.relu=nn.ReLU(inplace=True)
        self.cca = CCALayerwn(wn,channel=n_feat, reduction=16)
        self.esa= ESAwn(wn,num_feat=n_feat)
        self.sigmoid=nn.Sigmoid()
        # CCAL
    def forward(self,x):
        h1=self.hffb1(x)
        # h1_1=self.conv2(h1)
        h2=self.hffb2(h1)
        # h2_1=self.conv3(h2)
        h3=self.hffb3(h2)
        # h4=self.hffb4(h3)
        # h5=self.hffb5(h4)
        # h=torch.cat([h1,h2,h3,h4,h5],dim=1)
        h=torch.cat([h1,h2,h3],dim=1)
        out=self.conv1(h)
        out=self.relu(out)
        # out=self.conv4(out)
        # out=self.cca(out)
        cca_out=self.cca(out)
        esa_out=self.esa(out)
        out=out*(self.sigmoid(esa_out+cca_out))
        return out+x

# m2snet
class CNN1(nn.Module):
    def __init__(self,wn,in_channel,out_channel,map_size,pad,group):
        super(CNN1,self).__init__()
        # self.weight = nn.Parameter(torch.ones(channel,channel,map_size,map_size),requires_grad=False).cuda()
        # self.bias = nn.Parameter(torch.zeros(channel),requires_grad=False).cuda()
        # self.pad = pad
        # self.map_size=map_size
        self.conv=wn(nn.Conv2d(in_channel,out_channel,kernel_size=map_size,padding=pad,stride=1,groups=group))
        # self.norm = nn.BatchNorm2d(channel)
        self.prelu = nn.PReLU()

    def forward(self,x):
        # out = F.conv2d(x,self.weight,self.bias,stride=1,padding=self.pad)
        out=self.conv(x)
        # out = self.norm(out)
        out = self.prelu(out)
        return out
       
#Frequency Enhancement (FE) Operation 002
class FE(nn.Module):
    def __init__(self,
                 wn, in_channels, channels):
        super(FE, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.k2 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        self.k3 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        self.k4 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        # self.conv_3 = CNN1(wn,in_channels,3,1)
        # self.conv_5 = CNN1(wn,in_channels,5,2)
        # self.conv1=wn(nn.Conv2d(in_channels,in_channels,1))
    def forward(self, x):
        h1 = F.interpolate(self.pool(x), (x.size(-2), x.size(-1)), mode='nearest')
        # h1_map1=self.conv_3(h1)
        # h1_map2=self.conv_5(h1)
        # x_map1=self.conv_3(x)
        # x_map2=self.conv_5(x)
        h2 = x - h1
        # h2=self.conv1(abs(x-h1)+abs(x_map1-h1_map1)+abs(x_map2-h1_map2))
        F2 = torch.sigmoid(torch.add(self.k2(h2), x))
        out = torch.mul(self.k3(x), F2)
        out = self.k4(out)
        return out
    
    #008
class MSFEBlock(nn.Module):
    def __init__(self,wn,
                  in_channels, channels):
        super(MSFEBlock, self).__init__()
        
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        # self.conv_1 = CNN1(wn,in_channels,in_channels,1,0)
        self.conv_3 = CNN1(wn,in_channels,in_channels,map_size=3,pad=1,group=in_channels)
        self.conv_5 = CNN1(wn,in_channels,in_channels,map_size=5,pad=2,group=in_channels)
        self.k2 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        self.k3 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        self.k4 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        self.conv1=wn(nn.Conv2d(in_channels,channels,1))
    def forward(self, x):
        h1 = F.interpolate(self.pool(x), (x.size(-2), x.size(-1)), mode='nearest')
        # h1_1=self.conv_1(h1)
        h1_map1=self.conv_3(h1)
        h1_map2=self.conv_5(h1)
        # x_1=self.conv_1(x)
        x_map1=self.conv_3(x)
        x_map2=self.conv_5(x)      
        h2=self.conv1(abs(x-h1)+abs(x_map1-h1_map1)+abs(x_map2-h1_map2))
        F2 = torch.sigmoid(torch.add(self.k2(h2), x))
        out = torch.mul(self.k3(x), F2)
        out = self.k4(out)
        return out
    # 008
class HFFBlock(nn.Module):
    def __init__(self,wn,n_feat):
        super(HFFBlock,self).__init__()
        self.channels = int(n_feat // 2)
        self.channel = int(self.channels//2)
        self.path_1 = wn(nn.Conv2d(n_feat, self.channels, kernel_size=1, bias=False))
        self.path_2 = wn(nn.Conv2d(n_feat, self.channels, kernel_size=1, bias=False))
        self.path_1_1=wn(nn.Conv2d(self.channels, self.channel, kernel_size=1, bias=False))
        self.path_1_2 = wn(nn.Conv2d(self.channels, self.channel, kernel_size=1, bias=False))
        self.relu=nn.ReLU(inplace=True)
        self.cca=CCALayerwn(wn,self.channel,reduction=4)
        # 高频分支
        self.HConv = MSFEBlock(wn,self.channels ,self.channels )
        self.conv1 = wn(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False))
    def forward(self,x):
           #Low-Frequency Path
        path_1 = self.path_1(x)
        path_1 = self.relu(path_1)
        path_1_1=self.path_1_1(path_1)
        path_1_2=self.path_1_2(path_1)
        path_1_1=self.cca(path_1_1)
        path_1=torch.cat([path_1_1,path_1_2],dim=1)
        #High-Frequency Path
        path_2 = self.path_2(x)
        path_2 = self.relu(path_2)
        path_2 = self.HConv(path_2)
        output = self.conv1(torch.cat([path_1, path_2], dim=1))
        output = output + x
        return output
    #    002
class HFFB(nn.Module):
    def __init__(self,wn,n_feat):
        super(HFFB,self).__init__()
        channels = n_feat // 2
        self.channels_3 = channels // 2
        self.path_1 = wn(nn.Conv2d(n_feat, channels, kernel_size=1, bias=False))
        self.path_2 = wn(nn.Conv2d(n_feat, channels, kernel_size=1, bias=False))
        self.relu = nn.ReLU(inplace=True)
        # self.k1 = wn(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.dwconv = wn(nn.Conv2d(self.channels_3,self.channels_3,kernel_size=3,stride=1,padding=1,bias=False,groups=self.channels_3))
        self.pwconv = wn(nn.Conv2d(self.channels_3,self.channels_3,kernel_size=1))
        self.conv = wn(nn.Conv2d(self.channels_3,self.channels_3,kernel_size=3,stride=1,padding=1,bias=False))
        self.cca=CCAwn(wn,self.channels_3,reduction=4)
        # 高频分支
        self.HConv = FE(wn,channels ,channels )
        self.conv1 = wn(nn.Conv2d(channels*2, n_feat, kernel_size=1, bias=False))
    def forward(self,x):
    #Low-Frequency Path
        path_1 = self.path_1(x)
        path_1 = self.relu(path_1)
        path_3 , path_4 = torch.split(path_1,[self.channels_3,self.channels_3],dim=1)
        path_3=self.pwconv(self.dwconv(path_3))
        path_3=self.conv(path_3)
        path_3=self.cca(path_3)
        path_1=torch.cat([path_3,path_4],dim=1)
        # path_1 = self.k1(path_1)
        path_1  = self.relu(path_1)
        #High-Frequency Path
        path_2 = self.path_2(x)
        path_2 = self.relu(path_2)
        path_2 = self.HConv(path_2)
        path_2 = self.relu(path_2)
        output = self.conv1(torch.cat([path_1, path_2], dim=1))
        output = output + x
        return output 
       
    
@ARCH_REGISTRY.register()
class HFFNETwn(nn.Module):
    def __init__(self, num_in_ch=3,n_feats=48,upscale=2,conv= default_conv):
        super(HFFNETwn, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        kernel_size = 3
        scale = upscale
        n_feat = n_feats
        self.c = int(n_feat//4)
        
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
     
        self.head = conv(wn,num_in_ch, n_feat, kernel_size)      
        self.convtrans=wn(torch.nn.ConvTranspose2d(num_in_ch, num_in_ch, 2,stride=2))
        self.vfconv=wn(nn.Conv2d(num_in_ch,n_feat,1))
        # self.ftb=FTB(n_feat)
        # 06
        self.fab=FABlockwn(wn,n_feat)
        self.conv2=wn(nn.Conv2d(n_feat*2,n_feat,1))
        self.lffb1=LFFB(wn,n_feat)
        self.lffb2=LFFB(wn,n_feat)
        self.lffb3=LFFB(wn,n_feat)
        self.lffb4=LFFB(wn,n_feat)
        self.lffb5=LFFB(wn,n_feat)
        self.lffb6=LFFB(wn,n_feat)
        self.relu=nn.ReLU(inplace=True)
        self.conv1=wn(nn.Conv2d(n_feat*6,n_feat,1))
        # self.conv1=wn(nn.Conv2d(n_feat*12,n_feat,1))
        self.conv3=wn(nn.Conv2d(n_feat,n_feat,kernel_size=3,padding=1,stride=1,bias=False))
        # self.conv=default_conv(n_feat,n_feat,1)

        modules_tail = [
            conv(wn,n_feat, scale * scale * 3, kernel_size=3),
            nn.PixelShuffle(scale),
        ]
        self.tail = nn.Sequential(*modules_tail)

    
    def forward(self,x,vf):
        (H, W) = (x.shape[2], x.shape[3])
        x=self.sub_mean(x)
        
        y_input = self.head(x)
        res=y_input
        vf=vf.permute(0,3,1,2)
        
        vf=self.relu(self.convtrans(self.convtrans(self.relu(self.convtrans(self.convtrans(vf))))))
        vf=self.sub_mean(vf)
        vf=self.vfconv(vf)
        # vf=self.ftb(vf)
        vf=self.fab(vf)
        y=torch.cat((vf,y_input),1)
        y=self.conv2(y)
        
        l1=self.lffb1(y)
        l2=self.lffb2(l1)
        l3=self.lffb3(l2)
        l4=self.lffb4(l3)
        l5=self.lffb5(l4)
        l6=self.lffb6(l5)
        out=self.conv1(torch.cat([l1,l2,l3,l4,l5,l6],dim=1))
        out=self.conv3(out)+res
        # output=self.conv(out)+res
        output = self.tail(out)
        output=self.add_mean(output)
        # vf=self.tail(vf)
        # vf=self.add_mean(vf)
        return output,vf

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
