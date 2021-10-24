import torch
import torch.nn as nn
import inspect
import sys
sys.path.append("../")
from utils import MODELS
def build_optim(model,regiter,**arg):
    t = register.get(args.pop('type'))
    return t(model.parameters(),**arg)

for module_name in dir(torch.optim):
    _optim = getattr(torch.optim, module_name)
    if inspect.isclass(_optim) and issubclass(_optim,torch.optim.Optimizer):
        MODELS.register_module()(_optim)