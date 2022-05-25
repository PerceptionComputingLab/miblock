import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
import socket
from datetime import datetime
import sys

sys.path.append('../')
from .registry import MODELS, OPTIMIZER, LOSS, build
from datasets import build_dataset, build_dataloader
from miseg import build_optim


class Runner:
    def __init__(self, cfg):
        print(cfg)
        self.train_dataset = build_dataset(cfg['data_process_pipeline'],cfg["train_data"]["image_dir"],
                                           cfg["train_data"]["label_dir"],cfg["mode"])
        self.checked = cfg["checked"] and self.train_dataset.check_size()
        if self.checked:
            cfg["loader"]["batch_size"] = 1
        self.train_dataloader = build_dataloader(self.train_dataset, cfg["loader"])
        self.model = build(cfg["model"], MODELS)
        self.optimizer = build_optim(cfg["optimizer"], OPTIMIZER, self.model)
        self.loss = build(cfg["loss"], LOSS)
        self.epochs = cfg["epochs"]
        self.name = cfg["model"]
        self.logdir = ""
        # logdir is the path of training process record file
        if "logdir" in cfg:
            self.logdir = cfg["logdir"]
        else:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.logdir = os.path.join(os.getcwd(), current_time + '_' + socket.gethostname())
        self.boardWriter = SummaryWriter(self.logdir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if "pretrain" in cfg:
            self.pretrain = cfg["pretrain"]
        else:
            self.pretrain = ""
        # self.hook =cfg["hook"]
        self.model = self.model.cuda()
        self.loss = self.loss.cuda()

    def call_hook(self, event_name, *args):
        args = args
        event = self.hook[event_name]
        event = build(event, args)
        event()

    def run(self):
        for i in range(1, self.epochs + 1):
            self.train()

    def iteration(self, model, mode="train"):
        loss_epoch = 0
        for i, (images, labels) in enumerate(self.train_dataloader):
            # print("loss", loss_epoch)
            images, labels = images.type(torch.FloatTensor), labels
            # print("images_shape", images.shape)
            # print("labels_shape", labels.shape)
            images, labels = images.cuda(), labels.cuda()
            _, _, C, W, H = images.size()
            if self.checked:
                for i in range(0, C):
                    image = images[:, :, i, ...]
                    label = labels[:, :, i, ...]

                    # print("image_shape", image.shape)
                    # print("label_shape", label.shape)
                    out = model(image)
                    out = torch.unsqueeze(out, 2)
                    # todo
                    # out = torch.argmax(out, dim=1)
                    # loss = []
                    # if isinstance(out, tuple):
                    #     for output in out:
                    #         loss.append(self.loss(output, y))
                    #     loss = loss.sum()
                    # else:
                    #     loss = self.loss(out, y)
                    #     print(loss)
                    # out = out[0, ...]
                    # label = label[0, ...]
                    # print("out_size", out.size())
                    # print("label_size", label.size())
                    # print("label", label[label == 1].sum(),label.dtype)
                    # print("out", out[out == 1].sum(),out.dtype)
                    loss = self.loss(label, out)
                    # print("loss",loss)
                    # if (labels.max() > 0):
                    #     print('i', i)
                    #     print('x_max', images.max())
                    #     print('x', images.min())
                    #     print('y_max', labels.max())
                    #     print('y', labels.min())
                    #     print('output_max', out.max())
                    #     print('output_min', out.min())
                    loss.requires_grad_(True)
                    loss_epoch += loss.item()
                    if mode == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    else:
                        pass
        return loss_epoch

    def train(self):
        model = self.model
        loss = 0
        if self.pretrain != "":
            pretrained_dict = torch.load(self.pretrain)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        for index in range(1, self.epochs + 1):
            scheduler = StepLR(self.optimizer, step_size=1e-5, gamma=0.1)
            loss = self.iteration(model, "train")
            scheduler.step()
            print("epoch:", index, "  loss:", loss)
            self.boardWriter.add_scalar(f"Loss", loss, index)
        torch.save(model.state_dict(), os.path.join(self.logdir, f"{self.name}_{loss}.pt"))

    def test(self):
        model = self.model

    def eval(self):
        model = self.model
