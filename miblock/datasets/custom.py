import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from os import listdir
from .pipeline import Compose,DATASETS
@DATASETS.register_module()
class CustomDataset(Dataset):
    def __init__(self,pipeline,file_dir,img_dir,lab_dir,mode):
        self.pipeline = Compose(pipeline)
        self.file_dir = file_dir
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.mode = mode
        self.img_infos = self.loadfiles(self.file_dir,self.img_dir,self.lab_dir)
        

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, index):
        print(self.img_infos[index])
        return self.pipeline(self.img_infos[index])

    def loadfiles(self,file_dir,img_dir,lab_dir):
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
    dataset = CustomDataset(cfg['train_pipeline'],cfg['file_dir'],cfg['img_dir'],cfg['lab_dir'],cfg['mode'])
    return dataset
