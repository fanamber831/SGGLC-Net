
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
from basicsr.archs.hffn_arch import *

class SE_Block(nn.Module):
    def __init__(self,wn, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            wn(nn.Linear(ch_in, ch_in // reduction, bias=False)),
            nn.ReLU(inplace=True),
            wn(nn.Linear(ch_in // reduction, ch_in, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class FeaBlock(nn.Module):
    def __init__(self,wn, C_in, C_out):
        super(FeaBlock, self).__init__()
        self.conv1 = wn(nn.Conv2d(C_in, C_out, 1, 1, 0))
        # self.bn = nn.BatchNorm2d(C_out)
        self.act = nn.PReLU()
        self.attention = SE_Block(wn,C_out, reduction=8)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn(out)
        out = self.act(out)
        out = self.attention(out)
        return out

class PartialConv(nn.Module):
    def __init__(self,wn, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        groups=in_channels
        self.input_conv = wn(nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias))
        # self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
        #                            stride, padding, dilation, groups, False)

        # torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        # for param in self.mask_conv.parameters():
        #     param.requires_grad = False

    def forward(self, inputt):
        # http://masc.cs.gmu.edu/wiki/partialconv

        input = inputt

        output = self.input_conv(input)
        out = output
  
        return out

class PCBActiv(nn.Module):
    def __init__(self, wn,in_ch, out_ch, bn=False, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(wn,in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(wn,in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(wn,in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(wn,in_ch, out_ch, 3, 1, 1, bias=conv_bias)

      
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            # out = self.bn(out)
            out = self.activation(out)
            out = self.conv(out)
            # out = self.bn(out)
            out = self.activation(out)

        elif self.innorm:
            out = self.conv(out)
            # out = self.bn(out)
            out = self.activation(out)
        elif self.outer:
            out = self.conv(out)
            # out = self.bn(out)
        else:
            out = self.conv(out)
            # out = self.bn(out)
            if hasattr(self, 'activation'):
                out = self.activation(out)
        return out

class ConvDown(nn.Module):
    def __init__(self,wn, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False, layers=1, activ=True):
        super().__init__()
        nf_mult = 1
        nums = out_c / 64
        sequence = []

        for i in range(1, layers + 1):
            nf_mult_prev = nf_mult
            if nums == 8:
                if in_c == 512:

                    nfmult = 1
                else:
                    nf_mult = 2

            else:
                nf_mult = min(2 ** i, 8)
            if kernel != 1:

                if activ == False and layers == 1:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c)
                    ]
                else:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c),
                        nn.LeakyReLU(0.2, True)
                    ]

            else:

                sequence += [
                    nn.Conv2d(in_c, out_c,
                              kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                    nn.InstanceNorm2d(out_c),
                    nn.LeakyReLU(0.2, True)
                ]

            if activ == False:
                if i + 1 == layers:
                    if layers == 2:
                        sequence += [
                            nn.Conv2d(nf_mult * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    else:
                        sequence += [
                            nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    break

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)