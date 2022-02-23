import sys
sys.path.append('../')
from .registry import PIPELINE,MODELS,OPTIMIZER,LOSS,build
from datasets import build_dataset,build_dataloader
from miseg import build_model,build_optim
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import socket
from datetime import datetime

class Runner:
    def __init__(self,cfg):
        self.dataset = build_dataset(cfg)
        self.dataloader = build_dataloader(self.dataset,cfg["loader"])
        self.model = build_model(cfg["model"],MODELS)
        self.optimizer = build_optim(cfg["optimizer"],OPTIMIZER,self.model)
        self.loss = build_model(cfg["loss"],LOSS)
        self.epochs = cfg["epochs"]
        if "logdir" in cfg:
            self.logdir = cfg["logdir"]
            self.boardWriter = SummaryWriter(self.logdir)
        else:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.logdir = os.path.join(os.getcwd(), current_time + '_' + socket.gethostname())
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if "pretrain" in cfg:
            self.pretrain = cfg["pretrain"]
        else:
            self.pretrain = ""
        #self.hook =cfg["hook"]
        self.model = self.model.cuda()
        self.loss = self.loss.cuda()


    def call_hook(self, event_name, *args):
        args = args
        event = self.hook[event_name]
        event = build(event)
        event()

    def run(self):
        for i in range(1, self.epochs + 1):
            self.train()
            #self.call_hook('epoch', i)

    def iteration(self, model, mode="train"):
        loss_epoch = 0
        for i, (x, y) in enumerate(self.dataloader):
            print("loss",loss_epoch)
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = []
            if isinstance(out, tuple):
                for output in out:
                    loss.append(self.loss(output, y))
                loss = loss.sum()
            else:
                loss = self.loss(out, y)
                print(loss)
            loss_epoch += loss.item()
            if mode == "train":
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
            else:
                pass
            # self.call_hook('iteration',x, y, *hook_data)
        return loss_epoch

    def train(self):
        model = self.model
        if self.pretrain != "":
            pretrained_dict = torch.load(self.pretrain)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        for index in range(1, self.epochs + 1):
            loss = self.iteration(model, "train")
            self.boardWriter.add_scalar(f"Loss", loss, index)