# CE loss and Focal loss
import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # input: (N, C), target: (N,). C is the number of classes
        logpt = -F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(logpt)
        loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
