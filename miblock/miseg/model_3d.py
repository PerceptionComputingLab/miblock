import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("../")
from utils import MODELS


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out, kernel_size=2, stride=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


@MODELS.register_module()
class SegNet_3d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SegNet_3d, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv3d(in_channel, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm3d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm3d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm3d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm3d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm3d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm3d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm3d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm3d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm3d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm3d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm3d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm3d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm3d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm3d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm3d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm3d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm3d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm3d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm3d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm3d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm3d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm3d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm3d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm3d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm3d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv3d(64, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d


@MODELS.register_module()
class Attention_UNet_3d(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(Attention_UNet_3d, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channel, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


@MODELS.register_module()
class UNet_3d(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, drop_rate=0.2, bilinear=True):
        super(UNet_3d, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channel, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


@MODELS.register_module()
class ResUNet_3d(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, drop_rate=0.2, bilinear=True):
        super(ResUNet_3d, self).__init__()

        self.Down1 = down_conv(ch_in=16, ch_out=32)
        self.Conv1 = conv_block(ch_in=in_channel, ch_out=16)

        self.Down2 = down_conv(ch_in=32, ch_out=64)
        self.Conv2 = conv_block(ch_in=32, ch_out=32)

        self.Down3 = down_conv(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=64, ch_out=64)

        self.Down4 = down_conv(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=128, ch_out=128)

        self.Conv5 = conv_block(ch_in=256, ch_out=256)

        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        f1 = self.Conv1(x)

        x2 = self.Down1(f1)
        f2 = self.Conv2(x2) + x2

        x3 = self.Down2(f2)
        f3 = self.Conv3(x3) + x3

        x4 = self.Down3(f3)
        f4 = self.Conv4(x4) + x4

        x5 = self.Down4(f4)
        f5 = self.Conv5(x5) + x5

        d5 = self.Up5(f5)
        e5 = torch.cat((f4, d5), dim=1)
        g5 = self.Up_conv5(e5) + d5

        d4 = self.Up4(g5)
        e4 = torch.cat((f3, d4), dim=1)
        g4 = self.Up_conv4(e4) + d4

        d3 = self.Up3(g4)
        e3 = torch.cat((f2, d3), dim=1)
        g3 = self.Up_conv3(e3) + d3

        d2 = self.Up2(g3)
        e2 = torch.cat((f1, d2), dim=1)
        g2 = self.Up_conv2(e2) + d2

        d1 = self.Conv_1x1(g2)

        return d1


def passthrough(x, **kwargs):
    return x

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 0)
        out = self.relu1(torch.add(out, x16))
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


@MODELS.register_module()
class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, upsample='trilinear'):
        super(OutConvBlock, self).__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=1, padding=0))

        if upsample == 'transpose':
            temp = stride // 2
            while temp != 1:
                # 每次上采样步长为2
                ops.append(nn.ConvTranspose3d(n_filters_out, n_filters_out, kernel_size=1, stride=2, padding=0))
                temp //= 2
            ops.append(nn.ConvTranspose3d(n_filters_out, n_filters_out, kernel_size=1, stride=2, padding=0))
            ops.append(nn.Conv3d(n_filters_out, n_filters_out, kernel_size=1, padding=0))
        elif upsample == 'trilinear':
            temp = stride // 2
            while temp != 1:
                # 每次上采样步长为2
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
                temp //= 2
            ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            ops.append(nn.Conv3d(n_filters_out, n_filters_out, kernel_size=1, padding=0))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConvBlockfor1(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(OutConvBlockfor1, self).__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=1, stride=1, padding=0))
        ops.append(nn.Conv3d(n_filters_out, n_filters_out, kernel_size=1, stride=1, padding=0))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvAttentionBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, ratio):
        super(ConvAttentionBlock, self).__init__()

        n_filters_cut = n_filters_in // ratio
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_cut, 3, padding=1))
        ops.append(nn.BatchNorm3d(n_filters_cut))
        ops.append(nn.ReLU(inplace=True))
        ops.append(nn.Conv3d(n_filters_cut, n_filters_out, 1, padding=0))
        ops.append(nn.Sigmoid())
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        y = self.conv(x)
        # res add
        return x * y + x


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


@MODELS.register_module()
class VNetMultiHead(nn.Module):
    """
    https://www.researchgate.net/publication/
    359947471_Uncertainty-Guided_Symmetric_Multi-Level_Supervision_Network_for_3D_Left_Atrium_Segmentation_in_Late_Gadolinium-Enhanced_MRI
    """
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', ratio=8, has_att=False,
                 has_enout=True, has_dropout=False):
        super(VNetMultiHead, self).__init__()
        self.has_dropout = has_dropout
        self.has_att = has_att
        self.has_enout = has_enout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)
        # auxiliary prediction, before downsampling
        self.block_one_cab = ConvAttentionBlock(n_filters, n_filters, ratio=ratio)
        self.block_one_out = OutConvBlockfor1(n_filters,
                                              n_classes)  # OutConvBlock(n_filters, n_classes, stride=1, upsample='transpose')#

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)
        # auxiliary prediction, before downsampling
        self.block_two_cab = ConvAttentionBlock(n_filters * 2, n_filters * 2, ratio=ratio)
        self.block_two_out = OutConvBlock(n_filters * 2, n_classes, stride=2, upsample='trilinear')

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
        # auxiliary prediction, before downsampling
        self.block_three_cab = ConvAttentionBlock(n_filters * 4, n_filters * 4, ratio=ratio)
        self.block_three_out = OutConvBlock(n_filters * 4, n_classes, stride=4, upsample='trilinear')

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)
        # auxiliary prediction, before downsampling
        self.block_four_cab = ConvAttentionBlock(n_filters * 8, n_filters * 8, ratio=ratio)
        self.block_four_out = OutConvBlock(n_filters * 8, n_classes, stride=8, upsample='trilinear')

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
        # auxiliary prediction, before upsampling
        self.block_six_cab = ConvAttentionBlock(n_filters * 8, n_filters * 8, ratio=ratio)
        self.block_six_out = OutConvBlock(n_filters * 8, n_classes, stride=8, upsample='trilinear')

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
        # auxiliary prediction, before upsampling
        self.block_seven_cab = ConvAttentionBlock(n_filters * 4, n_filters * 4, ratio=ratio)
        self.block_seven_out = OutConvBlock(n_filters * 4, n_classes, stride=4, upsample='trilinear')

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
        # auxiliary prediction, before upsampling
        self.block_eight_cab = ConvAttentionBlock(n_filters * 2, n_filters * 2, ratio=ratio)
        self.block_eight_out = OutConvBlock(n_filters * 2, n_classes, stride=2, upsample='trilinear')

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_cab = ConvAttentionBlock(n_filters, n_filters, ratio=ratio)
        self.block_nine_out = OutConvBlockfor1(n_filters,
                                               n_classes)  # OutConvBlock(n_filters, n_classes, stride=1, upsample='transpose')#

        self.logits_out = nn.Conv3d(n_classes * 4, n_classes, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):

        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)
        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        # 输出
        if self.has_att:
            x6 = self.block_six_cab(x6)
            x7 = self.block_seven_cab(x7)
            x8 = self.block_eight_cab(x8)
            x9 = self.block_nine_cab(x9)
        x6_out = self.block_six_out(x6)
        x7_out = self.block_seven_out(x7)
        x8_out = self.block_eight_out(x8)
        x9_out = self.block_nine_out(x9)

        decoder_out = torch.cat((x6_out, x7_out, x8_out, x9_out), 1)
        decoder_out_logits = self.logits_out(decoder_out)
        # out_logits = self.logits_out(x9)
        # out_dis = self.dis_out(x9)

        return decoder_out_logits

    def encoder_out(self, features):
        # 输出
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        if self.has_att:
            x1 = self.block_one_cab(x1)
            x2 = self.block_two_cab(x2)
            x3 = self.block_three_cab(x3)
            x4 = self.block_four_cab(x4)
        x1_out = self.block_one_out(x1)
        x2_out = self.block_two_out(x2)
        x3_out = self.block_three_out(x3)
        x4_out = self.block_four_out(x4)
        encoder_out = torch.cat((x1_out, x2_out, x3_out, x4_out), 1)
        encoder_out_logits = self.logits_out(encoder_out)
        return encoder_out_logits

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)

        decoder_out_logits = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        if self.has_enout:
            encoder_out_logits = self.encoder_out(features)
            return decoder_out_logits, encoder_out_logits
        else:
            return decoder_out_logits