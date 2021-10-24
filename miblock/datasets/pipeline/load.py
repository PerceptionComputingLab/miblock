import SimpleITK as sitk
import torch
import numpy as np
import sys
sys.path.append('../../')
from utils import PIPELINE
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
        """
        if isinstance(path,tuple):
            img = sitk.ReadImage(path[0])
            img_array = sitk.GetArrayFromImage(img)

            lab = sitk.ReadImage(path[1])
            lab_array = sitk.GetArrayFromImage(lab)
         
            return img_array,lab_array
        else:
            img = sitk.ReadImage(path[0])
            img_array = sitk.GetArrayFromImage(img)
            #img_array = torch.FloatTensor(img_array.astype(np.float32))
            return img_array




