
from tkinter import N
from tkinter.font import nametofont
from turtle import forward
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from basicsr.archs.rcan_arch import *
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import Upsample, make_layer
from einops import rearrange
from basicsr.archs.arch_util import default_init_weights


@ARCH_REGISTRY.register()
class WyNet(nn.Module):
    def __init__(self, num_in_ch=3,n_feats=48,upscale=2, num_heads=8,conv= default_conv, norm_layer=nn.LayerNorm):
        super(LBNet, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        kernel_size = 3
        scale = upscale
        n_feat = n_feats
        num_head = num_heads
        ffn_expansion_factor=2.66
        bias=False

        # self.head = conv(num_in_ch, n_feat, kernel_size)
        self.fe=FE(num_in_ch,n_feats)
        d=[[[2,4],[4,8],[8,16]],[[4,8],[8,16],[16,32]]]
       
        self.msdbg1=MSDBGroup(n_feat,d=d[0])
        self.msdbg2=MSDBGroup(n_feat,d=d[0])
        self.msdbg3=MSDBGroup(n_feat,d=d[0])
        self.msdbg4=MSDBGroup(n_feat,d=d[1])
        # self.mab=mab(n_feat)
        self.nafg1=NAFGroup(n_feats=n_feat)
        self.nafg2=NAFGroup(n_feats=n_feat)

        self.conv=default_conv(n_feat,n_feat,1)

        modules_tail = [
            conv(n_feat, scale * scale * 3, 3),
            nn.PixelShuffle(scale),
        ]
        self.tail = nn.Sequential(*modules_tail)

    # def forward(self, x, vgg_feature):
    def forward(self,x):
        (H, W) = (x.shape[2], x.shape[3])
        # y_input = self.head(x)
        y_input=self.fe(x)
        res = y_input
       
        x2=self.msdbg1(y_input)
        # print("msdbg1:",x2)
        x3=self.msdbg2(x2)
        x4=self.msdbg3(x3)
        x5=self.msdbg4(x4)
        x6=self.nafg1(x5)
        x6=self.nafg2(x6)
        output=self.conv(x6)+res
        # y7=torch.cat([y1,y2,y3],dim=1)
        # y8 = self.c1(y7)

        output = self.tail(output)
        # return output,vgg_feature
        return output

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
