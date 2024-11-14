
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
from basicsr.archs.omnisr import OSA_Block, OSAm_Block
from basicsr.archs.shift import sconv
# lkdn
class Attentionlka(nn.Module):

    def __init__(self, dim):
        super(Attentionlka,self).__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn

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
        self.conv1 = wn(conv(dim, dim, kernel_size, bias=True))
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = wn(conv(dim, dim, kernel_size, bias=True))
        self.calayer = CCALwn(wn=wn,channel=dim,reduction=reduction)
        self.palayer = PALayerwn(wn,dim,reduction=reduction)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res = res + x
        return res  
    

    
class LFFB(nn.Module):
    def __init__(self,wn,n_feat):
        super(LFFB,self).__init__()
        self.norm = nn.LayerNorm(n_feat)
        self.hffb1=HFFBlock(wn,n_feat)
        self.hffb2=HFFBlock(wn,n_feat)
        self.hffb3=HFFBlock(wn,n_feat)
        default_init_weights([self.norm], 0.1)       
        self.conv1=wn(nn.Conv2d(n_feat*3,n_feat,1))
        self.relu=nn.ReLU(inplace=True)
        self.cca = CCALayerwn(wn,channel=n_feat, reduction=16)
        self.esa= ESAwn(wn,num_feat=n_feat)
        self.sigmoid=nn.Sigmoid()
        # CCAL
    def forward(self,x):
        res=x
        x=self.norm(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        h1=self.hffb1(x)
        h2=self.hffb2(h1)
        h3=self.hffb3(h2)
        h=torch.cat([h1,h2,h3],dim=1)
        out=self.conv1(h)
        out=self.relu(out)
        cca_out=self.cca(out)
        esa_out=self.esa(out)
        out=out*(self.sigmoid(esa_out+cca_out))
        return out+res
    
class LFFBm(nn.Module):
    def __init__(self,wn,n_feat):
        super(LFFBm,self).__init__()
        self.norm = nn.LayerNorm(n_feat)
        self.hffb1=HFFBmlock(wn,n_feat)
        self.hffb2=HFFBmlock(wn,n_feat)
        self.hffb3=HFFBmlock(wn,n_feat)
        default_init_weights([self.norm], 0.1)
        
        self.conv1=wn(nn.Conv2d(n_feat*3,n_feat,1))
        self.relu=nn.ReLU(inplace=True)
        self.cca = CCALayerwn(wn,channel=n_feat, reduction=16)
        self.esa= ESAwn(wn,num_feat=n_feat)
        self.sigmoid=nn.Sigmoid()
        # CCAL
    def forward(self,x):
        res=x
        x=self.norm(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        h1=self.hffb1(x)
        h2=self.hffb2(h1)
        h3=self.hffb3(h2)
        h=torch.cat([h1,h2,h3],dim=1)
        out=self.conv1(h)
        out=self.relu(out)
        cca_out=self.cca(out)
        esa_out=self.esa(out)
        out=out*(self.sigmoid(esa_out+cca_out))
        return out+res
# m2snet
class CNN1(nn.Module):
    def __init__(self,wn,in_channel,out_channel,map_size,pad,group):
        super(CNN1,self).__init__()
        self.conv=wn(nn.Conv2d(in_channel,out_channel,kernel_size=map_size,padding=pad,stride=1,groups=group))
        self.prelu = nn.PReLU()

    def forward(self,x):
        out=self.conv(x)
        out = self.prelu(out)
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
    
class MSFEBlock2(nn.Module):
    def __init__(self,wn,
                  in_channels, channels):
        super(MSFEBlock2, self).__init__()
        self.c=int(in_channels//3)
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        # self.conv_1 = CNN1(wn,in_channels,in_channels,1,0)
        self.conv_3 = CNN1(wn,in_channels,in_channels,map_size=3,pad=1,group=in_channels)
        self.conv_5 = CNN1(wn,in_channels,in_channels,map_size=5,pad=2,group=in_channels)
        # self.k2 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        # self.k3 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        self.k2=sconv(wn,in_channels,channels)
        self.k3=sconv(wn,in_channels,channels)
        self.k4 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        # self.k4=sconv(wn,in_channels,channels)
        self.conv1=wn(nn.Conv2d(in_channels,channels,1))
    def forward(self, x):
        h1 = F.interpolate(self.pool(x), (x.size(-2), x.size(-1)), mode='nearest')
        # h1_1=self.conv_1(h1)
        h1_map1=self.conv_3(h1)
        h1_map2=self.conv_5(h1)
        # x_1=self.conv_1(x)
        x_map1=self.conv_3(x)
        x_map2=self.conv_5(x)      
        h2=self.conv1(abs(x-h1)+abs(x_map1-h1_map1)+abs(h1_map2-x_map2))
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
        # self.cca=CCALayerwn(wn,self.channel,reduction=4)
        self.osa=OSA_Block(wn,self.channel)
        # self.osam=OSAm_Block(wn,self.channel)
        # 高频分支
        self.HConv = MSFEBlock2(wn,self.channels ,self.channels )
        self.conv1 = wn(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False))
    def forward(self,x):
           #Low-Frequency Path
        path_1 = self.path_1(x)
        path_1 = self.relu(path_1)
        path_1_1=self.path_1_1(path_1)
        path_1_2=self.path_1_2(path_1)
        path_1_1=self.osa(path_1_1)
        path_1=torch.cat([path_1_1,path_1_2],dim=1)
        #High-Frequency Path
        path_2 = self.path_2(x)
        path_2 = self.relu(path_2)
        path_2 = self.HConv(path_2)
        output = self.conv1(torch.cat([path_1, path_2], dim=1))
        output = output + x
        return output
class HFFBmlock(nn.Module):
    def __init__(self,wn,n_feat):
        super(HFFBmlock,self).__init__()
        self.channels = int(n_feat // 2)
        self.channel = int(self.channels//2)
        self.path_1 = wn(nn.Conv2d(n_feat, self.channels, kernel_size=1, bias=False))
        self.path_2 = wn(nn.Conv2d(n_feat, self.channels, kernel_size=1, bias=False))
        self.path_1_1=wn(nn.Conv2d(self.channels, self.channel, kernel_size=1, bias=False))
        self.path_1_2 = wn(nn.Conv2d(self.channels, self.channel, kernel_size=1, bias=False))
        self.relu=nn.ReLU(inplace=True)
        # self.cca=CCALayerwn(wn,self.channel,reduction=4)
        # self.osa=OSA_Block(wn,self.channel)
        self.osam=OSAm_Block(wn,self.channel)
        # 高频分支
        self.HConv = MSFEBlock2(wn,self.channels ,self.channels )
        self.conv1 = wn(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False))
    def forward(self,x):
           #Low-Frequency Path
        path_1 = self.path_1(x)
        path_1 = self.relu(path_1)
        path_1_1=self.path_1_1(path_1)
        path_1_2=self.path_1_2(path_1)
        path_1_1=self.osam(path_1_1)
        path_1=torch.cat([path_1_1,path_1_2],dim=1)
        #High-Frequency Path
        path_2 = self.path_2(x)
        path_2 = self.relu(path_2)
        path_2 = self.HConv(path_2)
        
        output = self.conv1(torch.cat([path_1, path_2], dim=1))
        output = output + x
        return output
    
    
@ARCH_REGISTRY.register()
class HFFNETwnosamsptvgg3(nn.Module):
    def __init__(self, num_in_ch=3,n_feats=48,upscale=2,conv= default_conv):
        super(HFFNETwnosamsptvgg3, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        kernel_size = 3
        scale = upscale
        n_feat = n_feats
        # self.c = int(n_feat//4)
        # div2k
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        # df2k
        # self.sub_mean = MeanShift((0.4690, 0.4490, 0.4036), sub=True)
        # self.add_mean = MeanShift((0.4690, 0.4490, 0.4036), sub=False)
        self.head = conv(wn,num_in_ch, n_feat, kernel_size)      
        self.convtrans=wn(torch.nn.ConvTranspose2d(num_in_ch, num_in_ch, 2,stride=2))
        self.vfconv=wn(nn.Conv2d(num_in_ch,n_feat,1))
        self.fab=FABlockwn(wn,n_feat)
        self.conv2=wn(nn.Conv2d(n_feat*2,n_feat,1))
        self.conv4=wn(nn.Conv2d(n_feat*2,n_feat,1))
        self.conv5=wn(nn.Conv2d(n_feat*2,n_feat,1))
        self.conv6=wn(nn.Conv2d(n_feat*2,n_feat,1))
        self.lffb1=LFFBm(wn,n_feat)
        self.lffb2=LFFBm(wn,n_feat)
        self.lffb3=LFFBm(wn,n_feat)
        self.lffb4=LFFB(wn,n_feat)
        self.lffb5=LFFB(wn,n_feat)
        self.lffb6=LFFB(wn,n_feat)
        self.relu=nn.ReLU(inplace=True)
        self.conv1=wn(nn.Conv2d(n_feat*6,n_feat,1))
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
        vf=self.relu(self.convtrans(self.relu(self.convtrans(self.convtrans(vf)))))
        vf=self.sub_mean(vf)
        vf=self.vfconv(vf)
        y1=torch.cat((vf,y_input),1)
        y1=self.conv2(y1)
        p1=self.fab(y1)     
        l1=self.lffb1(y_input)
        l2=self.lffb2(l1)
        lv1=self.fab(self.conv4(torch.cat((l2,vf),dim=1)))
        f1=p1*lv1+lv1
        l3=self.lffb3(f1)
        l4=self.lffb4(l3)
        p2=p1+self.fab(p1)
        lv2=self.fab(self.conv5(torch.cat((l4,vf),dim=1)))
        f2=p2*lv2+lv2
        l5=self.lffb5(f2)
        l6=self.lffb6(l5)
        out=self.conv1(torch.cat([l1,l2,l3,l4,l5,l6],dim=1))
        out=self.conv3(out)+res
        output = self.tail(out)
        output=self.add_mean(output)
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
