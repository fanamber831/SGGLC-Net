from optparse import NO_DEFAULT
from tkinter import N
from tkinter.font import nametofont
from turtle import forward
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import Upsample, make_layer
from einops import rearrange
from timm.models.layers.helpers import to_2tuple
from basicsr.archs.arch_util import default_init_weights
import torchvision.models as models

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


@ARCH_REGISTRY.register()
class RCAN(nn.Module):
    """Residual Channel Attention Networks.

    ``Paper: Image Super-Resolution Using Very Deep Residual Channel Attention Networks``

    Reference: https://github.com/yulunzhang/RCAN

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(RCAN, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x


def make_model(args, parent=False):
    return LBNet(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = self.sigmoid(x)
        return y * res

# BSNR引入ESA和CCA
def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x*y

class ESA(nn.Module):
    def __init__(self, num_feat=32, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = num_feat // 4
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, f, 1)
        self.conv5 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1 = (self.conv1(input))
        c2 = self.conv2(c1)
        pool = self.maxPooling(c1)
        c3 = self.GELU(self.conv3(pool))
        # c3 = self.GELU(self.conv3(v_range))
        # c3 = self.conv3_(c3)
        up = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        c4 = self.conv4(c1)
        c5 = self.conv5((up + c4))
        m = self.sigmoid(c5)

        return input * m

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class AIEB(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(AIEB, self).__init__()
        self.conv3_1_A = nn.Conv2d(in_channels, out_channels, (1, 3), stride=1, padding=(0, 1))
        self.conv3_1_B = nn.Conv2d(in_channels, out_channels, (3, 1), stride=1, padding=(1, 0))
        self.relu_1 = nn.LeakyReLU(inplace=True)
        self.conv3_2_A = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_2_B = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
    def forward(self, input):
        x1 = self.conv3_1_A(input)
        x2= self.conv3_1_B(input)
        x3 = self.relu_1(x1)
        x4= self.relu_1(x2)
        x5= self.conv3_2_A(x3)
        x6= self.conv3_2_B(x4)
        x7 = x5 + x6
        return x7

# 浅层特征提取时尝试分层提取
class FE(nn.Module):
    def __init__(self,
                 in_channels, n_feats,
                 ksize=3, stride=1, pad=1):
        super(FE, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, n_feats//4, 1, 1, 0),
            nn.ReLU(inplace=True))
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, n_feats//4, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats//4, n_feats//4, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats//4, n_feats//4, ksize, stride, pad),
            nn.ReLU(inplace=True))
        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels, n_feats//4, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats//4, n_feats//4, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        self.body4 = nn.Sequential(
            nn.Conv2d(in_channels, n_feats//4, ksize, stride, pad),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        out3 = self.body3(x)
        out4 = self.body4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

#  multi-scale feature extraction
class MSBlock(nn.Module):
    def __init__(self, n_feats=32):
        super(MSBlock,self).__init__()
        distill_rate=0.25
        self.distilled_channels = int(n_feats*distill_rate)
        self.c1=nn.Conv2d(n_feats,n_feats//4,kernel_size=1)
        self.c2=nn.Conv2d(n_feats,n_feats//4,kernel_size=3,padding=2,dilation=2)
        self.c3=nn.Conv2d(n_feats//2,n_feats//4,kernel_size=1)
        self.c4=nn.Conv2d(n_feats,n_feats//4,kernel_size=3,padding=4,dilation=4)
        self.c5=nn.Conv2d(n_feats//2,n_feats//4,kernel_size=1)
        self.c6=nn.Conv2d(n_feats,n_feats//4,kernel_size=3,padding=8,dilation=8)
        self.c7=nn.Conv2d(n_feats//2,n_feats//4,kernel_size=1)
        self.c8=nn.Conv2d(n_feats,n_feats,kernel_size=3,padding=1)
        self.c9=nn.Conv2d(n_feats,n_feats,kernel_size=3,padding=1)
        self.c10=nn.Conv2d(n_feats,n_feats,kernel_size=3,padding=1)
        self.c11=nn.Conv2d(n_feats*2,n_feats,kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.se = CCALayer(channel=n_feats, reduction=16)
        self.sa = ESA()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        res=x
        channel1,channel2,channel3,channel4 = torch.split(x,(self.distilled_channels,self.distilled_channels,self.distilled_channels,self.distilled_channels),dim=1)
        y1=self.act(self.c1(x))
        y2=torch.cat([y1,self.act(self.c2(x))],dim=1)
        y3=self.act(self.c3(y2))
        y4=torch.cat([y3,self.act(self.c4(x))],dim=1)
        y5=self.act(self.c5(y4))
        y6=torch.cat([y5,self.act(self.c6(x))],dim=1)
        y7=self.act(self.c7(y6))
        y8=self.act(self.c8(x))
        y9=self.act(self.c9(y8))
        y10=self.act(self.c10(y9))
        y11=torch.cat([y1,y3,y5,y7,y10],dim=1)
        y12=self.c11(y11)
        sa_out=self.se(y12)
        ca_out=self.sa(y12)
        y=y12*(self.sigmoid(sa_out+ca_out))
        return y

# restormer.trans
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False,drop_out_rate=0.):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.conv1 = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.sg=SimpleGate()
        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        input=x
        x = self.conv1(x)
        x= self.sg(x)
        x = self.conv2(x)
        x=self.dropout(x)
        return input+x*self.gamma


## Multi-DConv Head Transposed Self-Attention (MDTA) restormer
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv=nn.Conv2d(dim,dim,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        # channel shuffle pixel-wise attention
        res=out
        out=out.view(b,2,c//2,h,w)
        out=torch.transpose(out,1,2).contiguous()
        out=out.view(b,-1,h,w)
        out=res*(self.sigmoid(self.conv(out)))
        return out
# dwrseg:sir module
class SIR(nn.Module):
    def __init__(self,dim):
        super(SIR,self).__init__()
        self.conv1=nn.Conv2d(dim,dim*3,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(dim*3,dim,kernel_size=1)
        self.act=nn.ReLU(inplace=True)
    def forward(self,x):
        res=x
        y1=self.act(self.conv1(x))
        y2=self.conv2(y1)
        return res+y2
# nafnet
class naf(nn.Module):
    def __init__(self, dim, bias,drop_out_rate=0.):
        super(naf, self).__init__()
        self.conv=nn.Conv2d(dim*2,dim,kernel_size=3,padding=1,groups=dim)
        self.qkv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.fbconv = FasterBlock(dim=dim,n_div=4, forward="split_cat", kernel_size=3,  padding=1 ,dil=1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv2=nn.Conv2d(dim,dim,kernel_size=1,bias=bias)
        self.conv3=nn.Conv2d(dim//2,dim//2,1)
        self.conv4=nn.Conv2d(dim//2,dim,1)
        self.conv5=nn.Conv2d(dim,dim,1)
        self.sg=SimpleGate()
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim//2, out_channels=dim// 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dropout= nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv(x)
        x1,x2 = qkv.chunk(2, dim=1)
        x1=self.fbconv(self.conv1(x1))
        x2=self.conv2(x2)
        out=x1*x2
        out=self.conv5(out)
        # out=self.conv(qkv)
        out=self.sg(out)
        # channel shuffle pixel-wise attention
        # res=out
        # out=out.view(b,2,c//4,h,w)
        # out=torch.transpose(out,1,2).contiguous()
        # out=out.view(b,-1,h,w)
        # out=res*(self.conv3(out))
        out=out*self.sca(out)
        out=self.conv4(out)
        out=self.dropout(out)
        out=x+out*self.beta
        return out

class NAFBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(NAFBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = naf(dim, bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim,ffn_expansion_factor,bias)

        default_init_weights([self.norm1], 0.1)
        default_init_weights([self.norm2], 0.1)
    def forward(self, x):

        x = x + self.attn(self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous())
        x = x + self.ffn(self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous())

        return x

class NAFGroup(nn.Module):
    def __init__(self,n_feats):
        super(NAFGroup,self).__init__()
        self.naf1=NAFBlock(dim=n_feats,ffn_expansion_factor=2.66,bias=False)
        self.naf2=NAFBlock(dim=n_feats,ffn_expansion_factor=2.66,bias=False)
        self.naf3=NAFBlock(dim=n_feats,ffn_expansion_factor=2.66,bias=False)
        self.naf4=NAFBlock(dim=n_feats,ffn_expansion_factor=2.66,bias=False)
        self.conv=nn.Conv2d(n_feats,n_feats,3,1,1)
    def forward(self,x):
        res=x
        x=self.conv(self.naf4(self.naf3(self.naf2(self.naf1(x)))))
        return x+res

class SIRGroup(nn.Module):
    def __init__(self,n_feats):
        super(SIRGroup,self).__init__()
        self.sir1=SIR(dim=n_feats)
        self.sir2=SIR(dim=n_feats)
        self.sir3=SIR(dim=n_feats)
        self.sir4=SIR(dim=n_feats)
        self.conv=nn.Conv2d(n_feats,n_feats,3,1,1)
    def forward(self,x):
        res=x
        x=self.conv(self.sir4(self.sir3(self.sir2(self.sir1(x)))))
        return x+res

class LGBGroup(nn.Module):
    def __init__(self,n_feats,d=[1,3,5]):
        super(LGBGroup,self).__init__()
        self.lgb1=LFFM(n_feats=n_feats,d=d)
        self.lgb2=LFFM(n_feats=n_feats,d=d)
        self.lgb3=LFFM(n_feats=n_feats,d=d)
        self.lgb4=LFFM(n_feats=n_feats,d=d)
        self.conv=nn.Conv2d(n_feats,n_feats,3,1,1)
    def forward(self,x):
        res=x
        x=self.conv(self.lgb4(self.lgb3(self.lgb2(self.lgb1(x)))))
        return x+res

class MSDBGroup(nn.Module):
    def __init__(self,n_feats):
        super(MSDBGroup,self).__init__()
        self.msd1=MSDB(n_feats)
        self.msd2=MSDB(n_feats)
        self.msd3=MSDB(n_feats)
        self.conv=nn.Conv2d(n_feats,n_feats,3,1,1)
    def forward(self,x):
        res=x
        x=self.conv(self.msd3(self.msd2(self.msd1(x))))
        return x+res

class LFFM(nn.Module):
    def __init__(self, n_feats=32, d=[1,3,5]):
        super(LFFM, self).__init__()
        # self.b1=MSBlock(n_feats=n_feats)
        # self.b2=MSBlock(n_feats=n_feats)
        # self.b3=MSBlock(n_feats=n_feats)
        # self.c1 = nn.Conv2d(2 * n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        # self.c2 = nn.Conv2d(3 * n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        # self.c3 = nn.Conv2d(4 * n_feats, n_feats, 1, stride=1, padding=0, groups=1)
        self.msd1=MSDB(n_feats,d=d[0])
        self.msd2=MSDB(n_feats,d=d[1])
        self.msd3=MSDB(n_feats,d=d[2])
        self.act=nn.ReLU(inplace=True)
# change
# 组卷积
        self.c1 = nn.Conv2d(n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        self.c2 = nn.Conv2d(n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        self.c3 = nn.Conv2d(3 * n_feats, n_feats, 1, stride=1, padding=0, groups=1)
    def forward(self, x):
        res = x
        out1=self.msd1(x)
        out2=self.msd2(self.c1(out1))
        out3=self.msd3(self.c2(out2))
        cat1=torch.cat([out1,out2,out3],1)
        out4=self.c3(cat1)
        output = res + out4
        return output

class LGBlock(nn.Module):
    def __init__(self,n_feats,d=[1,3,5]):
        super(LGBlock,self).__init__()
        self.r=LFFM(n_feats=n_feats,d=d)
        self.naf=NAFBlock(dim=n_feats,ffn_expansion_factor=2.66,bias=False)
        self.c=nn.Conv2d(in_channels=n_feats,out_channels=n_feats,kernel_size=1)
    def forward(self,x):
        res=x
        x=self.c(self.naf(self.r(x)))
        return res+x

class SIRBlock(nn.Module):
    def __init__(self,n_feats):
        super(SIRBlock,self).__init__()
        self.s=SIR(dim=n_feats)
        self.naf=NAFBlock(dim=n_feats,ffn_expansion_factor=2.66,bias=False)
        self.c=nn.Conv2d(n_feats,n_feats,kernel_size=1)
    def forward(self,x):
        res=x
        x=self.c(self.naf(self.s(x)))
        return res+x

# FBSNet
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
# MSDB
class M(nn.Module):
    def __init__(self,n_feat,d=[2,4]):
        super(M,self).__init__()
        self.dis_c=int(n_feat/3)
        self.c1=nn.Conv2d(self.dis_c,self.dis_c,kernel_size=1)
        self.c2=nn.Conv2d(self.dis_c*3,n_feat,1)
        self.pwconv1=nn.Conv2d(self.dis_c,self.dis_c,1)
        self.pwconv2=nn.Conv2d(self.dis_c,self.dis_c,1)
        self.c3=FasterBlock(dim=self.dis_c,n_div=4,forward="split_cat",kernel_size=3,padding=1,dil=1)
        self.c4=FasterBlock(dim=self.dis_c,n_div=4,forward="split_cat",kernel_size=3,padding=1,dil=1)
        self.c5=nn.Conv2d(n_feat,self.dis_c,kernel_size=1)
        self.gc=nn.Conv2d(self.dis_c,self.dis_c,kernel_size=3,stride=1,padding=1,groups=4)
        self.fc1 = nn.Sequential(FasterBlock(dim=self.dis_c,n_div=4,forward="split_cat", kernel_size=3 , padding= 1,dil=1))
        self.fc2_1 = FasterBlock(dim=self.dis_c,n_div=4,forward="split_cat",kernel_size=(3,1),padding=(1*d[0], 0),dil=(d[0], 1))
        self.fc2_2 = FasterBlock(dim=self.dis_c, n_div=4,forward="split_cat",kernel_size=(1,3), padding=(0, 1*d[0]),dil=(1, d[0]))
        self.fc4_1 = FasterBlock(dim=self.dis_c,n_div=4,forward="split_cat", kernel_size=(3,1), padding=(1*d[1], 0),dil=(d[1], 1))
        self.fc4_2 = FasterBlock(dim=self.dis_c, n_div=4,forward="split_cat",kernel_size=(1,3), padding=(0, 1*d[1]),dil=(1, d[1]))
        self.act = nn.PReLU()
        self.cca1 = CCALayer(channel=n_feat, reduction=16)
        self.esa= ESA()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        channel1,channel2,channel3 = torch.split(x,(self.dis_c,self.dis_c,self.dis_c),dim=1)
        x1=self.c1(channel1)
        x2=self.gc(channel2)
        x3_0=self.fc1(channel3)
        x3=self.act(self.fc2_1(x3_0)+x3_0)
        x3=self.fc2_2(x3)+x3
        x3=self.pwconv1(x3)
        x4=self.act(self.fc4_1(x3_0)+x3_0)
        x4=self.fc4_2(x4)+x4
        x4=self.pwconv2(x4)
        x5=torch.cat([x3_0,x3,x4],dim=1)
        x5=self.c5(x5)
        x11=x2+self.c3(x1)
        x12=x5+self.c4(x11)
        cat=torch.cat([x1,x11,x12],dim=1)
        x=self.c2(cat)
        cca_out=self.cca1(x)
        esa_out=self.esa(x)
        out=x*(self.sigmoid(esa_out+cca_out))
        return out

class MSDB(nn.Module):
    def __init__(self,n_feat):
        super(MSDB,self).__init__()
        self.dis_c=int(n_feat*0.125)
        self.n_feat=n_feat
        self.dwconv_hw = nn.Conv2d(self.dis_c, self.dis_c, 3, padding=3//2, groups=self.dis_c)
        self.dwconv_w = nn.Conv2d(self.dis_c, self.dis_c, kernel_size=(1, 11), padding=(0, 11//2), groups=self.dis_c)
        self.dwconv_h = nn.Conv2d(self.dis_c, self.dis_c, kernel_size=(11, 1), padding=(11//2, 0), groups=self.dis_c)
        self.c1=nn.Conv2d(self.n_feat,self.n_feat,kernel_size=1)
        self.act = nn.PReLU()
        self.cca = CCALayer(channel=n_feat, reduction=16)
        self.esa= ESA()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        channel1,channel2,channel3,channel4 = torch.split(x,(self.dis_c,self.dis_c,self.dis_c,self.n_feat-self.dis_c*3),dim=1)
        x1=self.dwconv_hw(channel1)
        x2=self.dwconv_w(channel2)
        x3=self.dwconv_h(channel3)
        cat=torch.cat([x1,x2,x3,channel4],dim=1)
        x=self.c1(cat)
        cca_out=self.cca(x)
        esa_out=self.esa(x)
        out=x*(self.sigmoid(esa_out+cca_out))
        return out

class mab(nn.Module):
     def __init__(self,n_feat):
         super(mab,self).__init__()
         self.cca = CCALayer(channel=n_feat, reduction=16)
         self.esa= ESA()
     def forward(self,x):
         res=x
         cca=self.cca(x)
         esa=self.esa(x)
         return res+cca+esa
#sca
def _pdf(x):
    x_mean  = torch.mean(x,dim=(-2,-1),keepdim=True)
    x_bias  = x - x_mean
    # x_k1    =  x_bias.abs().mean(dim=(-2,-1),keepdim=True)
    x_k2 =  x_bias.pow(2).mean(dim=(-2,-1),keepdim=True)  # 二阶中心矩
    # x_k3 = x_bias.pow(3).mean(dim=(-2,-1),keepdim=True)
    x_k4  =  x_bias.pow(4).mean(dim=(-2,-1),keepdim=True)

    x_sigma = x_k2.pow(1/2)      # 为了统一量纲
    # # x_skew = x_k3 / x_sigma.pow(1.5)        # 偏度 开3次方
    x_kurt = x_k4.pow(1/4) / x_sigma       # 峰度 开4次方

    return torch.cat([x_mean,x_sigma,x_kurt,],dim=-1)

def _spatialpool(x):
    x_avg = torch.mean(x, dim=1,keepdim=True)
    x_max = torch.max( x, dim=1,keepdim=True)[0]

    return torch.cat([x_avg, x_max], dim=1)

## Channel Attention (CA) Layer  ----from RCAN
class SCALayer(nn.Module):
    def __init__(self, n_feats, bias=True, reduction=16, increase=2):
        super(SCALayer, self).__init__()

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.f = _pdf
        self.conv_du = nn.Sequential(
            nn.Conv2d(n_feats, 4, (1,1), padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, n_feats, (1,3), padding=0, bias=bias),
            nn.Sigmoid(),
        )
        # self.sptialpool = _spatialpool
        # self.conv_space = nn.Sequential(
        #     # nn.Conv2d(n_feats, 2*n_feats, 1, padding=0, bias=bias),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(1, 1, 3, padding=1, bias=bias),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(2, 1, 3, padding=1, bias=bias),
        #     nn.Sigmoid(),
        # )
        # self.out = nn.Conv2d(2*n_feats,n_feats,1,1,0,bias=bias )
        # self.coef = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0, 1.0]))


    def forward(self, x):
        # print(self.coef)
        y = self.f(x)
        y = self.conv_du(y)        #   N,C,1,1
        y = x*y

        # z = self.sptialpool(x)
        # z = self.conv_space(z) - 0.5
        # z = x*z*self.coef[3]
        # # out = self.out(torch.cat([y,z],dim=1))
        # t = self.coef[0] + self.coef[1]
        return y

def channel_shuffle(x,groups):
    b,c,h,w=x.shape
    x=x.reshape(b,groups,-1,h,w)
    x=x.permute(0,2,1,3,4)
    x=x.reshape(b,-1,h,w)
    return x

class PConv(nn.Module):
    # partial convolution
    def __init__(self,dim,n_div=4,forward="split_cat",kernel_size=3,padding=1,dil=1):
        super().__init__()
        # dim:输入/输出通道数 n_div:partial比例的倒数
        self.dim_conv=dim//n_div #除 取整数
        self.dim_untouched = dim - self.dim_conv
        self.conv=nn.Conv2d(self.dim_conv,self.dim_conv,kernel_size,stride=1,padding=padding,bias=False,dilation=dil)
        self.conv1=nn.Conv2d(self.dim_conv,self.dim_conv,1)
        self.act=nn.PReLU()
        # if forward == "slicing":
        #     self.forward = self.forward_slicing
        if forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    # def forward_slicing(self,x:Tensor)->Tensor:
    #     # apply forward pass for inference
    #     x[:,:self.dim_conv,:,:]=self.conv(x[:,:self.dim_conv,:,:])
    #     return x
    def forward_split_cat(self,x):
        # apply forward pass for training
        x1,x2=torch.split(x,[self.dim_conv,self.dim_untouched],dim=1)
        x1=self.conv(x1)
        x1=self.conv1(self.act(x1))
        x=torch.cat((x1,x2),1)
        x=channel_shuffle(x,4)
        return x
# pconv+pwconv
class FasterBlock(nn.Module):
    def __init__(self,dim,n_div,forward,kernel_size,padding,dil):
        super().__init__()
        self.pconv=PConv(dim,n_div,forward,kernel_size=kernel_size,padding=padding,dil=dil)
        self.pwconv1=nn.Conv2d(dim,dim*2,kernel_size=1,stride=1,padding=0)
        self.pwconv2=nn.Conv2d(dim*2,dim,kernel_size=1,stride=1,padding=0)
        self.act=nn.PReLU()
    def forward(self,x):
        res=x
        x=self.pconv(x)
        x=self.act(self.pwconv1(x))
        x=self.pwconv2(x)
        return x+res

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


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

class metaformblock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(metaformblock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = InceptionDWConv2d(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ConvMlp(dim)

        default_init_weights([self.norm1], 0.1)
        default_init_weights([self.norm2], 0.1)
    def forward(self, x):

        x = x + self.attn(self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous())
        x = x + self.ffn(self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous())

        return x

@ARCH_REGISTRY.register()
class LBNet(nn.Module):
    def __init__(self, num_in_ch=3,n_feats=32,upscale=2,conv= default_conv, norm_layer=nn.LayerNorm):
        super(LBNet, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        kernel_size = 3
        scale = upscale
        n_feat = n_feats
        ffn_expansion_factor=2.66
        bias=False
        self.convtrans=torch.nn.ConvTranspose2d(num_in_ch, num_in_ch, 2,stride=2)
        self.vfconv=nn.Conv2d(num_in_ch,n_feat,1)
        # self.head = conv(num_in_ch, n_feat, kernel_size)
        self.fe=FE(num_in_ch,n_feats)
        self.ftb=FTB(n_feat)
        self.conv2=nn.Conv2d(n_feat*2,n_feat,1)

        # d=[[[2,4],[4,8],[8,16]],[[4,8],[8,16],[16,32]]]
        self.msdbg1=MSDBGroup(n_feat)
        self.msdbg2=MSDBGroup(n_feat)
        self.msdbg3=MSDBGroup(n_feat)
        self.msdbg4=MSDBGroup(n_feat)
        self.msdbg5=MSDBGroup(n_feat)
        self.msdbg6=MSDBGroup(n_feat)

        # self.mab=mab(n_feat)
        self.nafg1=NAFGroup(n_feats=n_feat)
        self.nafg2=NAFGroup(n_feats=n_feat)
        self.nafg3=NAFGroup(n_feats=n_feat)
        self.nafg4=NAFGroup(n_feats=n_feat)
        self.sir1=SIRGroup(n_feats=n_feat)
        self.sir2=SIRGroup(n_feats=n_feat)



        self.conv=default_conv(n_feat,n_feat,1)

        modules_tail = [
            conv(n_feat, scale * scale * 3, 3),
            nn.PixelShuffle(scale),
        ]
        self.tail = nn.Sequential(*modules_tail)

    # def forward(self, x, vgg_feature):
    def forward(self,x,vf):
        (H, W) = (x.shape[2], x.shape[3])
        # y_input = self.head(x)
        y_input=self.fe(x)
        res = y_input
        vf=vf.permute(0,3,1,2)
        vf=self.convtrans(self.convtrans(self.convtrans(self.convtrans(vf))))
        vf=self.vfconv(vf)
        vf=self.ftb(vf)
        y=torch.cat((vf,y_input),1)
        y=self.conv2(y)
        x2=self.sir1(y)
        x2=self.sir2(x2)
        x2=self.msdbg1(x2)
        x3=self.msdbg2(x2)
        x4=self.msdbg3(x3)
        x5=self.msdbg4(x4)
        x5=self.msdbg5(x5)
        x5=self.msdbg6(x5)
        x6=self.nafg1(x5)
        x6=self.nafg2(x6)
        x6=self.nafg3(x6)
        x6=self.nafg4(x6)
        output=self.conv(x6)+res

        output = self.tail(output)
        # return output,vgg_feature
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
