import torch
import torch.nn as nn
import inspect
import sys
sys.path.append("../")
from utils import OPTIMIZER

def build_optim(cfg,register,model):
    for module_name in dir(torch.optim):
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,torch.optim.Optimizer):
            OPTIMIZER.register_module()(_optim)
    args = cfg.copy()
    t = register.get(args.pop('type'))
    return t(model.parameters(),**args)
