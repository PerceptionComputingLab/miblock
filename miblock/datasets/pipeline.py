import sys
sys.path.append('../')
import SimpleITK as sitk
import torch
import numpy as np
import random
from utils import PIPELINE,build
@PIPELINE.register_module()
class Compose(object):
    """Compose  transforms together.

    Args:
        transforms: A list of transform class dict.
    """
    def __init__(self,transforms):
        self.transform = []
        for transform in transforms:
            self.transform.append(build(transform,PIPELINE))


    def __call__(self,data):
        """Make tansforms run sequentially.

        Args:
            data:Filenames of the data or dict that contain the informations of the data

        Returns:
           dict: Transformed data.
        """

        for t in self.transform:
            data = t(data)
            if data is None:
                return None
        return data


@PIPELINE.register_module()
class LoadImage(object):
    """Load a medical image

    """
    def __init__(self):
        pass
    def __call__(self,path):
        """Call functions to load image
        Args:
            path:The path of the image
            In the train mode,the path will be a tuple
            In the test mode,the path will be a string
        """
        if isinstance(path,tuple):
            img = self.func(path[0])
            lab = self.func(path[1])
            return img,lab
        else:
            img = self.func(path)
            return img

    def func(self,data):
        img = sitk.ReadImage(data)
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array.astype(np.float32)
        img_array = torch.FloatTensor(img_array).unsqueeze(0)
        return img_array

@PIPELINE.register_module()
class Normalization():
    def __init__(self,):
        pass
    def __call__(self,data):
        if isinstance(data,tuple):
            img = self.func(data[0])
            lab = self.func(data[1])
            return img,lab
        else:
            return self.func(data)
    def func(self,data):
        _, _, _, C = data.shape
        bg_mask = data == 0
        mean_arr = np.zeros(C, dtype="float32")
        std_arr = np.zeros(C, dtype="float32")
        for i in range(C):
            data = data[..., i]
            selected_data = data[data > 0]
            mean = np.mean(selected_data)
            std = np.std(selected_data)
            mean_arr[i] = mean
            std_arr[i] = std

        norm_volume = (data - mean_arr) / std_arr
        norm_volume[bg_mask] = 0
        return norm_volume

@PIPELINE.register_module()
class RandomCrop():
    """
    Args:
        crop_size:Matrix size after cropping
    """
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
class RandomResize():
    def __init__(self,size):
        self.size = size

    def __call__(self,data):
        pass