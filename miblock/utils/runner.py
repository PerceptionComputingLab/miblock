import sys
sys.path.append('../')
from .registry import PIPELINE,MODELS,OPTIMIZER,LOSS,build
from .config import Config
from datasets import build_dataset,build_dataloader
from miseg import build_model,build_optim

class Runner:
    def __init__(self,cfg):
        self.dataset = build_dataset(cfg)
        self.dataloader = build_dataloader(self.dataset,cfg["loader"])
        self.model = build_model(cfg["model"],MODELS)
        self.optimizer = build_optim(cfg["optimizer"],OPTIMIZER,self.model)
        self.loss = build_model(cfg["loss"],LOSS)
        #self.hook =cfg["hook"]
        print(self.model)
        print(self.optimizer)
        print(self.loss)
        
    def call_hook(self, event_name, *args):
        args = (time, ) + args
        event = self.hook[event_name]
        event = build(event)
        event()
    
    def run(self, epochs = 1):    
        for i in range(1, epochs + 1):
            self.train()
            #self.call_hook('epoch', i)
    
    def train(self):
        for i, (x, y) in enumerate(self.dataloader):   
            print(x.shape)
            print(y.shape)
            out = self.model(x)
            loss = self.loss(out, y)
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()
            
            #self.call_hook('iteration',x, y, *hook_data)
