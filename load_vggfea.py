import os
import glob
import cv2
import numpy as np

def get_img_path(img_dir):
    # print("img_dir:",img_dir)
    img_paths=glob.glob(os.path.join(img_dir,"*.png"))
    # img_paths=os.path.join(img_dir+"*.png")
    # print(img_paths)
    # file_list=[]
    # file_list.extend(glob.glob(img_dir+'*.png'))
    print(img_paths)
    # for img_path in img_paths:
    #     imgs=cv2.imread(img_path).astype(np.float32)
    #     # print("图片路径：")
    #     print(imgs)
    # imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_dir]
    # print("111")
        
if __name__=="__main__":
    img_dir="E:/srtest/Set5/GTmod12"
    get_img_path(img_dir)