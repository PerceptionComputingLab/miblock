import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
sys.path.append("../")
from utils import LOSS
def onehot(input, class_n):
    '''
    onehot for pytorch
    :param input: N*H*W
    :param class_n:
    :return:N*n_class*H*W
    '''
    shape = input.shape
    onehot = torch.zeros((class_n,) + shape).cuda()
    for i in range(class_n):
        onehot[i, ...] = (input == i)

    onehot_trans = onehot.permute(1, 0, 2, 3,4)
    return onehot_trans
@LOSS.register_module()
class CrossEntryLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,target,pred,alpha=[0.1,1]):
        '''
        calculate the croos-entropy loss function
        :param y_true: tensor, size=N*D*H*W
        :param y_pred: tensor, size=N*class_n*D*H*W
        :return: voxel weighted cross entropy loss
        '''
        log_prob = F.log_softmax(pred, dim=1)
        shape = pred.shape
        y_true_tensor = onehot(target, shape[1])
        loss = 0
        for i in range(shape[1]):
            y_task = y_true_tensor[:, i, ...]
            y_prob = log_prob[:, i, ...]
            loss += torch.mean(-y_task * y_prob) * alpha[i]
        return loss

### From https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
@LOSS.register_module()
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


### From https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
@LOSS.register_module()
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

@LOSS.register_module()
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


### From https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
@LOSS.register_module()
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU