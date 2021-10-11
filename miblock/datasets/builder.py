from .custom import CustomDataset
import sys
sys.path.append('../')
from utils import Registry
DATASETS = Registry('dataset')
PIPELINE = Registry('pipeline')
def build_dataset(cfg):
    dataset = CustomDataset(cfg['pipeline'],cfg['file_dir'],cfg['img_dir'],cfg['lab_dir'])
    return dataset


