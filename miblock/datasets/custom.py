import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from os import listdir
from .pipeline import Compose
from .builder import DATASETS
@DATASETS.register_module()
class CustomDataset(Dataset):
    def __init__(self,pipeline,file_dir,img_dir,lab_dir):
        self.pipeline = Compose(pipeline)
        self.file_dir = file_dir
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_infos = self.loadfiles(self.file_dir,self.img_dir,self.lab_dir)

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, index):
        return self.pipeline(img_infos[index])

    def loadfiles(self,file_dir,img_dir,lab_dir):
        if self.test:
            #img_dir = file_dir + img_dir
            img = listdir(img_dir)
            return sorted(img)
        else:
            #img_dir = file_dir + img_dir
            img = listdir(img_dir)
            #lab_dir = file_dir + lab_dir
            lab = listdir(lab_dir)
            return list(zip(sorted(img),sorted(lab)))
