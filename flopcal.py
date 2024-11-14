from basicsr.archs import hffn_arch
from ptflops import get_model_complexity_info
from tensorboard import summary
# from thop import get_model_complexity_info
import torch
def input_constructor():
    # 这里假设有两个输入，一个为 RGB 图像，大小为 224x224，另一个为单通道图像，大小为 128
    input_size_1 = (3, 320, 180)
    input_size_2 = (15, 15 , 3)
    x1 = torch.randn((1,) + input_size_1)
    x2 = torch.randn((1,) + input_size_2)
    return (x1, x2)
with torch.cuda.device(0):
    model = hffn_arch.HFFNET()
    # 定义两个输入参数
    x = torch.randn(1, 3, 320, 180)
    y = torch.randn(1, 15, 15, 3)

# 将两个参数的形状转换为整数类型，并打包成一个元组
    input_res = (tuple(map(int, x.shape)), tuple(map(int, y.shape)))
    # input_size=[(3,320,180),(15,15,3)]
    # input_res=((3,320,180),(15,15,3))
    get_model_complexity_info(model,input_res=input_res, as_strings=True, print_per_layer_stat=False, verbose=True)
# summary(model.cuda(),[[3,320,180],[15,15,3]],batch_size=48)