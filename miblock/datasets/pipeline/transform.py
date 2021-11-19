import sys
import random
import torch
sys.path.append('../../')
from utils import PIPELINE
@PIPELINE.register_module()
class Crop():
    def __init__(self,crop_size):
        self.crop_size = crop_size

    def __call__(self,data):
        size1 = random.randint(0,data[0].size(1) - self.crop_size )
        size2 = random.randint(0,data[0].size(2) - self.crop_size )
        size3 = random.randint(0,data[0].size(3) - self.crop_size )
        if isinstance(data,tuple):
            img_tensor = torch.zeros((data[0].size(0),\
                self.crop_size,self.crop_size,self.crop_size))
            lab_tensor = torch.zeros((data[1].size(0),\
                self.crop_size,self.crop_size,self.crop_size))
            img = data[0]
            lab = data[1]
            img_tensor = img[:, size1:size1+self.crop_size, \
                size2:size2+self.crop_size, size3:size3+self.crop_size]
            lab_tensor = lab[:, size1:size1+self.crop_size, \
                size2:size2+self.crop_size, size3:size3+self.crop_size]
            return img_tensor,lab_tensor
        else:
            img = data
            img_tensor = img[:, size1:self.crop_size, size2:self.crop_size, size3:self.crop_size]
            return img_tensor

@PIPELINE.register_module()
class Resize():
    def __init__(self,size):
        self.size = size

    def __call__(self,data):
        pass

