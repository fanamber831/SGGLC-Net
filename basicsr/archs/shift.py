from re import X
import torch
import torch.nn.functional as F
from torch import nn as nn

     
def spatial_shift(f, steps, pad):
   """
   f [torch.Tensor]: input feature in (B, C, H, W)
   steps [Tuple(Tuple(int, int))]: parameters of the spatial-shift steps
   pad [int]: padding size
   """
   shift_groups = len(steps)
   B, C, H, W = f.shape
   group_dim = C // shift_groups
   f_pad = F.pad(f, pad)
#    print(f_pad.shape)
   output = torch.zeros_like(f)
   for idx, step in enumerate(steps):
       s_h, s_w = step[0], step[1]
       # 从第idx*group_dim个通道开始，到第(idx+1)*group_dim个通道结束
       # 偏移量为s_h, s_w
       # 起始偏移位置为pad[0] pad[2]
       output[:, idx * group_dim: (idx + 1) * group_dim, :, :] = f_pad[:, idx * group_dim:(idx + 1)
                                                                       * group_dim, pad[0] + s_h:pad[0] + s_h + H, pad[2] + s_w:pad[2] + s_w + W]
   return output


    
class sconv(nn.Module):
   # 1*1conv 8channel indentity 16shift
   def __init__(self,wn,n_feat,out):
        super(sconv,self).__init__() 
        
      #   self.channels1=int(n_feat//3)
      #   self.channels2=int(n_feat//3*2)     
        self.conv1=wn(nn.Conv2d(n_feat,out,kernel_size=1))
        self.conv2=wn(nn.Conv2d(n_feat,out,kernel_size=1))
        self.act=nn.ReLU(inplace=True)

      #   self.conv3=wn(nn.Conv2d(n_feat,out,kernel_size=3,stride=1,padding=1))
   def forward(self,x):
       res=x
      #  path1,path2=torch.split(x,[self.channels1,self.channels2],dim=1)
       x=self.conv1(x)  
       out=spatial_shift(x,steps=(
          (0,0),(0,0),(0,0),(0,0),
         (-2,2),(0,2),(2,2),
         (-2,0),(2,0),(-2,-2),
         (0,-2),(2,-2)),pad=[2,2,2,2]) 
       out=self.act(out)
      #  out=torch.cat([path1,out],dim=1)
       out=self.conv2(out)
    
       return out+res

class sconv64(nn.Module):
   # 1*1conv 8channel indentity 16shift
   def __init__(self,wn,n_feat,out):
        super(sconv64,self).__init__() 
        
      #   self.channels1=int(n_feat//3)
      #   self.channels2=int(n_feat//3*2)     
        self.conv1=wn(nn.Conv2d(n_feat,out,kernel_size=1))
        self.conv2=wn(nn.Conv2d(n_feat,out,kernel_size=1))
        self.act=nn.ReLU(inplace=True)

      #   self.conv3=wn(nn.Conv2d(n_feat,out,kernel_size=3,stride=1,padding=1))
   def forward(self,x):
       res=x
      #  path1,path2=torch.split(x,[self.channels1,self.channels2],dim=1)
       x=self.conv1(x)  
       out=spatial_shift(x,steps=(
          (0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),
         (-2,2),(0,2),(2,2),
         (-2,0),(2,0),(-2,-2),
         (0,-2),(2,-2)),pad=[2,2,2,2]) 
       out=self.act(out)
      #  out=torch.cat([path1,out],dim=1)
       out=self.conv2(out)
    
       return out+res
    
# data=torch.randn(1,48,3,3)
# print(data)
# print(data.shape)
# wn = lambda x: torch.nn.utils.weight_norm(x)
# shift=sconv(wn,48,48)
# s=shift(data)
# print(s)
# print(s.shape)
    