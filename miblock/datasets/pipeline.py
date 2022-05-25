import sys

sys.path.append('../')
import SimpleITK as sitk
import torch
import numpy as np
import random
import os
from utils import PIPELINE, build


@PIPELINE.register_module()
class Compose(object):
    """Compose data preprocessing and augmentation together.

    Args:
        transforms: A list of transforms class dict.
    """

    def __init__(self, transforms):
        self.transform = []
        for transform in transforms:
            self.transform.append(build(transform, PIPELINE))

    def __call__(self, data):
        """Make transforms run sequentially.

        Args:
            data:Filenames of the data or dict that contain the information of the data

        Returns:
           array: Transformed data.
        """

        for t in self.transform:
            data = t(data)
            if data is None:
                return None
        return data


@PIPELINE.register_module()
class LoadImage:
    """Load medical image
    """

    def __init__(self):
        pass

    def __call__(self, path):
        """Call functions to load image
        Args:
            path:Path of images file or folder
                In the train mode,it should be a tuple of paths to images and labels
                In the test mode,it should be a string of path to image
                Format of path:
                ├── file_dir
                │   ├── img_dir
                │   │   │   ├── 1
                │   │   │   ├── 2
                │   │   │   ├── 3
                │   ├── ann_dir
                │   │   │   ├── 1
                │   │   │   ├── 2
                │   │   │   ├── 3
                or multimodal:
                ├── file_dir
                │   ├── img_dir
                │   │   │   ├── 1
                │   │   │   │   ├── 1.1
                │   │   │   │   ├── 1.2
                ...
                │   │   │   ├── 2
                │   │   │   │   ├── 2.1
                │   │   │   │   ├── 2.2
                ...
                │   ├── ann_dir
                │   │   │   ├── 1
                │   │   │   │   ├── 1.1
                │   │   │   │   ├── 1.2
                ...
                │   │   │   ├── 2
                │   │   │   │   ├── 2.1
                │   │   │   │   ├── 2.2
                ...
        Returns:
            array:An array of image data
        """
        if isinstance(path, tuple):
            image = self.func(path[0])
            label = self.func(path[1])
            return image, label
        else:
            image = self.func(path)
            return image

    def func(self, data):
        """Load images according to file path.
        Args:
            data:Path of images file or folder
        """
        if os.path.isfile(data):
            image = []
            image.append(self.load(data))
            img_array = np.array(image)
            return img_array
        elif os.path.isdir(data):
            img_list = os.listdir(data)
            image = []
            for img_dir in img_list:
                image.append(self.load(os.path.join(data, img_dir)))
            img_array = np.array(image)
            return img_array
        else:
            pass

    def load(self, data):
        image = sitk.ReadImage(data)
        img_array = sitk.GetArrayFromImage(image)
        img_array = img_array.astype(np.float32)
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0)
        return img_array


@PIPELINE.register_module()
class Normalization:
    def __init__(self, ):
        pass

    def __call__(self, data):
        if isinstance(data, tuple):
            image = self.func(data[0])
            label = data[1]
            return image, label
        else:
            return self.func(data)

    def func(self, data):
        _, size1, size2, size3 = data.shape
        bg_mask = data == 0
        for i in range(size1):
            data_i = data[:, i, :, :]
            selected_data = data_i[data_i > 0]
            # Array is not all zero
            if selected_data.size > 0:
                mean = np.mean(selected_data)
                std = np.std(selected_data)
                data[:, i, :, :] = (data[:, i, :, :] - mean) / std
        data[bg_mask] = 0
        return data


@PIPELINE.register_module()
class RandomCrop:
    """
    Args:
        crop_size:Matrix size after cropping
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, data):
        size1 = data[0].shape[1] - self.crop_size
        size2 = data[0].shape[2] - self.crop_size
        size3 = data[0].shape[3] - self.crop_size
        blank = torch.zeros(1, self.crop_size, self.crop_size, self.crop_size)
        if size1 <= 0:
            print("The image size is smaller than the crop size,return an array of 0.")
            if isinstance(data, tuple):
                return blank, blank
            else:
                return blank
        size1 = random.randint(0, size1)
        size2 = random.randint(0, size2)
        size3 = random.randint(0, size3)
        if isinstance(data, tuple):
            image = data[0]
            label = data[1]
            image = image[:, size1:size1 + self.crop_size, \
                    size2:size2 + self.crop_size, size3:size3 + self.crop_size]
            label = label[:, size1:size1 + self.crop_size, \
                    size2:size2 + self.crop_size, size3:size3 + self.crop_size]
            return image, label
        else:
            image = data
            image = image[:, size1:self.crop_size, size2:self.crop_size, size3:self.crop_size]
            return image


@PIPELINE.register_module()
class AdjustWindow:
    """
    Args:
        window width:Range of Hu values for target organ
        window level:Median Hu value of the target organ
    """

    def __init__(self, window_width, window_level):
        self.limits = [window_level - window_width / 2, window_level + window_width / 2]

    def __call__(self, data):
        if isinstance(data, tuple):
            image = data[0].astype(np.float64)
            label = data[1].astype("uint8")
            image = self.func(image)
            return image.astype("uint8"), label.astype("uint8")
        else:
            image = self.func(data)
            return image.astype("uint8")

    def func(self, data):
        graymax = float(self.limits[1])
        graymin = float(self.limits[0])
        delta = 1 / (graymax - graymin)
        gray = delta * data - graymin * delta
        graycut = np.maximum(0, np.minimum(gray, 1))
        out = (graycut * 255)
        return out


@PIPELINE.register_module()
class RandomResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        pass
