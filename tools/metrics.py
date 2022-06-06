import sys
import argparse

sys.path.append('../miblock/')
from utils import Runner, Config



def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    # if there are multiple config files,cfg is list,else cfg is dict
    runner = Runner(cfg)
    runner.test()


if __name__ == '__main__':
    main()
