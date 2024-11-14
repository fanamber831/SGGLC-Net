from torchstat import stat
from basicsr.models import build_model
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
from os import path as osp

def FLOP(root_path):
    opt,args=parse_options(root_path)
    model=build_model(opt)
    stat(model,(3,320,180))

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    FLOP(root_path)