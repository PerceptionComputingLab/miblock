import SimpleITK as sitk
import torch
import numpy as np
from .builder import PIPELINE
@PIPELINE.register_module()
class LoadImage(object):
    def __init__(self):
        pass
    def __call__(self,path):
        img = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(img)
        img_array = torch.FloatTensor(img_array.astype(np.float32))
        return img_array

@PIPELINE.register_module()
class LoadLabel(object):
    def __init__(self):
        pass
    def __call__(self,path):
        img = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(img)
        img_array = torch.FloatTensor(img_array.astype(np.float32))
        return img_array




