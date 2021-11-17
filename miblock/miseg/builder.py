import sys
sys.path.append('../../')
from utils import build
def build_model(cfg,register):
    return build(cfg,register)
