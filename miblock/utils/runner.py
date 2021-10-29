import sys
sys.path.append('../')
from utils import Config,PIPELINE,MODELS,build
from datasets import build_dataset
from models import build_model,build_optim
class Runner():
    def __init__(self,cfg):
        self.dataset = build_dataset(cfg)
        self.model = build_model(cfg["model"],MODELS)
        self.optimizer = build(self.model,MODELS,cfg["optimizer"])
        self.loss = build_model(cfg["loss"],MODELS)
        self.hook =cfg["hook"]

    def call_hook(self, event_name, *args):
        args = (time, ) + args
        event = self.hook[event_name]
        event = build(event)
        event()
    
    def run(self, epochs = 1):    
        for i in range(1, epochs + 1):
            self.train()
            self.call_hook('epoch', i)
    
    def train(self):
        for i, (x, y) in enumerate(self.dataset, self.iterations + 1):      
            def closure():
                out = self.model(x)
                loss = self.loss(out, y)
                # TODO 
                loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            
            self.call_hook('iteration',x, y, *hook_data)
        self.iterations += i
