import argparse
import sys
sys.path.append('../miblock/')
from utils import Config,MODELS,OPTIMIZER,LOSS,build
from miseg import build_model,build_optim
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.load(args.config)
model = build_model(cfg["model"],MODELS)
print(model)
loss = build_model(cfg["loss"],LOSS)
print(loss)
optim = build_optim(cfg["optimizer"],OPTIMIZER,model)
print(optim)