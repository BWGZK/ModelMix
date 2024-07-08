import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np


class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, predict, target):

        w = 1 / ((einsum("bcwh->bc", target).type(torch.float32) + 1e-10) ** 2)
        intersection = w * einsum("bcwh,bcwh->bc", predict, target)
        union = w * (einsum("bcwh->bc", predict) + einsum("bcwh->bc", target))
        loss = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        return loss.mean()


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, predict, target):

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = (predict * target).sum(1)
        den = predict.sum(1) + target.sum(1)

        loss = 1 - (2 * num + 1e-5) / (den + 1e-5)

        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        
        dice = BinaryDiceLoss()
        total_loss = 0

        for i in range(target.shape[1]):
            dice_loss = dice(predict[:, i], target[:, i])
            total_loss += dice_loss

        return total_loss/target.shape[1]

class WCELoss(nn.Module):
    def __init__(self):
        super(WCELoss, self).__init__()
    
    def weight_function(self, target):

        mask = torch.argmax(target, dim=1)
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        weights = []
        for i in range(mask.max()+1):
            voxels_i = [mask==i][0].sum().cpu().numpy()
            if voxels_i == 0:
                voxels_i = 1
            w_i = np.log(voxels_sum/voxels_i).astype(np.float32)
            weights.append(w_i)
        weights = torch.from_numpy(np.array(weights)).cuda()
        
        return weights

    def forward(self, predict, target):

        ce_loss = torch.mean(-target * torch.log(predict + 1e-10), dim=(0,2,3))
        weights = self.weight_function(target)
        if weights.shape[0] == ce_loss.shape[0]:
            loss = weights * ce_loss
            return loss.sum()
        else:
           loss = weights * ce_loss[0:weights.shape[0]]
        return loss.sum()

# class WCELoss(nn.Module):
#     def __init__(self):
#         super(WCELoss, self).__init__()
    
#     def weight_function(self, target):

#         mask = torch.argmax(target, dim=1)
#         voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
#         weights = []
#         for i in range(mask.max()+1):
#             voxels_i = [mask==i][0].sum().cpu().numpy()
#             w_i = np.log(voxels_sum/voxels_i).astype(np.float32)
#             weights.append(w_i)
#         weights = torch.from_numpy(np.array(weights)).cuda()
        
#         return weights

#     def forward(self, predict, target):

#         ce_loss = torch.mean(-target * torch.log(predict + 1e-10), dim=(0,2,3))
#         weights = self.weight_function(target)
#         loss = weights * ce_loss
        
#         return loss.sum()


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, predict, target):

        ce_loss = -target * torch.log(predict + 1e-10)
        
        return ce_loss.mean()


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

        self.dice_loss = DiceLoss()
        self.wce_loss = WCELoss()

    def forward(self, predict, target,dice=True):

        wce_loss = self.wce_loss(predict, target)
        if dice:
            dice_loss = self.dice_loss(predict, target)
            loss = wce_loss + dice_loss
        else:
            loss = wce_loss
        return loss
