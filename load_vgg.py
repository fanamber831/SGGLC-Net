import json
import os
import glob
from pyexpat import model
import cv2
import numpy as np
from basicsr.utils import img2tensor
import torchvision.models as models
import torch

def make_model():
    model=models.vgg16().cuda()	# 其实就是定位到第28层，对照着上面的key看就可以理解
    state_dict=torch.load('/home/lhk/basic/scripts/vgg16-397923af.pth')
    model.load_state_dict(state_dict)
    model=model.features[:28]
    return model

def get_img_path(model):
    model.eval()
    # print("img_dir:",img_dir)
    # img_paths=glob.glob(os.path.join(img_dir,"*.png"))
    conv=torch.nn.Conv2d(512,32,1)
    paths = ['/home/lhk/dataset/DF2K/DF2K_LRx2_sub','/home/lhk/dataset/testSRx2/Set5/LR_bicubic/X2',
             '/home/lhk/dataset/testSRx2/Set14/LR_bicubic/X2','/home/lhk/dataset/testSRx2/BSD100/LR_bicubic/X2',
             '/home/lhk/dataset/testSRx2/urban100/LR_bicubic/X2','/home/lhk/dataset/testSRx2/manga109/LR_bicubic/X2']
    data={}
    for i in paths: 
        img_paths=glob.glob(os.path.join(i,"*.png"))
        for img_path in img_paths:
            imgs=cv2.imread(img_path).astype(np.float32)/255
            imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
            imgs=imgs.cuda()
            c,h,w=imgs.shape
            imgs=imgs.reshape(1,c,h,w)
            result=model(imgs)
            result=conv(result.data.cpu())
            result_npy=result.data.cpu().numpy()
            data[img_path]=result_npy[0]
    np.savez("my_dict_df2k.npz",**data)
    return data



if __name__=="__main__":
    # img_dir="/home/lhk/dataset/DF2K/DF2K_LRx2_sub"
    # model=make_model()
    # tmp=get_img_path(model)
    # print(tmp)
    data=np.load("my_dict_df2k.npz")
    data=dict(data)
    # data.close()
    # print(type(data))
    # a=data['E:/srtest/Set5/GTmod12\\woman.png']
    b = data['/home/lhk/dataset/testSRx2/Set5/LR_bicubic/X2/woman.png']
    print(type(b))
    # print(data)