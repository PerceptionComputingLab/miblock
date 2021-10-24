import sys
sys.path.append('../../')
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from os import listdir
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
            mode (bool):if mode = True, enter test mode

    """
    def __init__(self,pipeline,img_dir,lab_dir,mode):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.mode = mode
        self.img_infos = self.loadfiles(self.img_dir,self.lab_dir)
        

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

    def loadfiles(self,img_dir,lab_dir):
        """Integrated data path.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            lab_dir (str): Path to label directory
        Returns:
            list[str]: All image info of dataset.
        """
        if self.mode == 'test':
            img = listdir(img_dir)
            return img
        else:
            img = listdir(img_dir)
            img = [img_dir + '/' + i for i in img]
            lab = listdir(lab_dir)
            lab = [lab_dir + '/' + i for i in lab]
            return list(zip(sorted(img),sorted(lab)))

def build_dataset(cfg):
    """Build the dataset
    """
    dataset = CustomDataset(cfg['train_pipeline'],cfg['img_dir'],cfg['lab_dir'],cfg['mode'])
    return dataset
