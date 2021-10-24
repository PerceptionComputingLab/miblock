import argparse
import sys
print(sys.path)
sys.path.append('../miblock/')
from utils import Config,MODELS,build
from models import build_model
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.load(args.config)
model = build_model(cfg["model"],MODELS)
print(model)
loss = build_model(cfg["loss"],MODELS)
print(loss)
optim = build_model(cfg["optimizer"],MODELS)