import sys
sys.path.append('../')
from utils import Config,PIPELINE,MODELS,build
from datasets import build_dataset
from models import build_model,build_optim
class Runner():
    def __init__(self,cfg):
        self.dataset = build_dataset(cfg)
        self.model = build_model(cfg["model"],MODELS)
        self.optim = build(self.model,MODELS,cfg["optimizer"])
        self.loss = build_model(cfg["loss"],MODELS)

    def train(self,):
        self.model.train()
        for epoch in range(cfg["num_epochs"]):
            for i, (imgs, masks) in enumerate(self.dataset):
                imgs = imgs.cuda()
                masks = masks.cuda()
                outputs = self.model(imgs)
                loss = self.loss(outputs, masks)
                self.optim .zero_grad()
                self.loss.backward()
                self.optim .step()
    def test():
        pass

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        for i, (imgs, masks) in enumerate(self.dataset):
            imgs = imgs.cuda()
            masks = masks.cuda()
            outputs = self.model(imgs)
            loss = criterion(outputs, masks)

