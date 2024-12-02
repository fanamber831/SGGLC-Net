# SGGLC-Net
This repo contains the code for our paper "Semantic-Guided Global-Local Collaborative Networks for Lightweight Image Super-Resolution''
![image](https://github.com/fanamber831/diyidaima/blob/main/fig1_911111%20(1).png)

### Dependencies
Please install following essential dependencies (see requirements.txt):
```
addict
future
lmdb
numpy>=1.17
opencv-python
Pillow
pyyaml
requests
scikit-image
scipy
tb-nightly
torch>=1.7
torchvision
tqdm
yapf
```

### Datasets and pre-processing
Download:  
1. **DIV2K**  (https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
2. **Set5**  (https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets)
3. **Set14**  (https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets)
4. **BSD100**  (https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
5. **Manga109**  (http://www.manga109.org/en/)

### Training  
python train.py -opt /rootdatal/basic/options/train/RCAN/train_HFFNETwnosamspt_x2.yml 

### Testing
python inference/inference_swinir.py --input datasets/Set5/LRbicx4 --patch_size 48 --model_path experiments/pretrained_models/SwinIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth --output results/SwinIR_SRX4_DIV2K/Set5


