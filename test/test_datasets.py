import argparse
import sys
sys.path.append('../miblock/')
from utils import Config,PIPELINE
from datasets import build_dataset
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.load(args.config)
assert 'img_dir' in cfg
pipeline = []
for p in cfg['train_pipeline']:
    pipeline.append(p['type'])
assert "LoadImage"  in pipeline
dataset = build_dataset(cfg)
print(dataset.__getitem__(1))
#print(PIPELINE.get('Compose'))
