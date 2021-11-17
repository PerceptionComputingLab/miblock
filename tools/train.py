import sys
sys.path.append('../miblock/')
from utils import Runner,Config
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.load(args.config)
    if isinstance(cfg,list):
        for cfgfile in cfg:
            runner = Runner(cfgfile)
            runner.train()
    else:
        runner = Runner(cfg)
        runner.train()
if __name__ == '__main__':
    main()