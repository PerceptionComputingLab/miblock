import sys

sys.path.append('../../')
from os import listdir
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .pipeline import Compose
from utils import DATASETS


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Basic dataset
        Format of data:
            ├── file_dir
            │   ├── img_dir
            │   │   │   ├── 1
            │   │   │   ├── 2
            │   │   │   ├── 3
            │   ├── ann_dir
            │   │   │   ├── 1
            │   │   │   ├── 2
            │   │   │   ├── 3
        The name of image and label needs to correspond

        Args:
            pipeline (list[dict]): Processing pipeline
            img_dir (str): Path to image directory
            lab_dir (str): Path to label directory
            mode (bool):if mode = "train", enter test mode

    """

    def __init__(self, pipeline, img_dir, lab_dir, mode="train",dir_map=False):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.mode = mode
        self.dir_map = dir_map
        self.img_infos = self.loadfiles(self.img_dir, self.lab_dir)
        #self.img_infos = self.img_infos[:100]

    def __len__(self):
        """Return the total number of data."""
        return len(self.img_infos)

    def __getitem__(self, index):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Transformed data.
        """
        return self.pipeline(self.img_infos[index])

    def check_size(self):
        """
        Only works when the data is 3d.
        If there is a large gap between the three dimensions of a certain data,
        for example, the data dimension is [30, 512, 512],512 is much larger than 30,
        or when the dimensions of several data are different and are not sliced, read in slice by slice;
        """
        if self.mode == "train":
            dim = self.__getitem__(0)[0].shape
            dim = np.array(dim[1:])
            dim_gap = ((1 / dim).reshape(3, 1)) @ (dim.reshape(1, 3))
            # for i in range(int(self.__len__())):
            #     dim2 = self.__getitem__(i)[0].shape
            #     if dim2[2]!=512 or dim2[3]!=512:
            #         self.img_infos.pop(i)
            #         print(dim2)
            #         print(self.img_infos[i])
            if dim_gap[dim_gap > 10].shape != (0,):
                return True
            for i in range(int(self.__len__())):
                dim2 = self.__getitem__(i)[0].shape
                print(dim2)
                dim2 = np.array(dim2[1:])
                if dim2[0] != dim[0] or dim2[1] != dim[1] or dim2[2] != dim[2]:
                    return True
                dim = dim2
            return False


    def loadfiles(self, img_dir, lab_dir):
        """Integrated data path.

        Args:
            img_dir (str): Path to image directory
            lab_dir (str): Path to label directory
        Returns:
            list[str]: All image info of dataset.
        """
        if self.dir_map:
            if self.mode == 'test':
                with open(img_dir, "r") as f:
                    img = f.read()
                    img = img.split()
                    return img
            else:
                with open(img_dir, "r") as f:
                    img = f.read()
                    img = img.split()
                with open(lab_dir, "r") as f:
                    lab = f.read()
                    lab = lab.split()
                return list(zip(sorted(img), sorted(lab)))
        else:
            if self.mode == 'test':
                img = listdir(img_dir)
                return img
            else:
                img = listdir(img_dir)
                img = [img_dir + '/' + i for i in img]
                lab = listdir(lab_dir)
                lab = [lab_dir + '/' + i for i in lab]
                return list(zip(sorted(img), sorted(lab)))


def build_dataset(pipeline, img_dir, lab_dir, mode):
    """Build the dataset
    """
    dataset = CustomDataset(pipeline, img_dir, lab_dir, mode)
    return dataset


def build_dataloader(dataset, cfg):
    dataloader = DataLoader(dataset=dataset, batch_size=cfg["batch_size"],
                            num_workers=cfg["num_workers"], shuffle=True)
    return dataloader
