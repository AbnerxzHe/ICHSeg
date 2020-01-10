import numpy as np
import torch

import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

# Borrows and modifies iou() from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions    
def iou(pred, target, num_classes):
    ious = np.zeros(num_classes)
    for cl in range(num_classes):
        pred_inds = (pred == cl)
        target_inds = (target == cl)
        intersection = pred_inds[target_inds].sum().to(torch.float32)
        union = pred_inds.sum() + target_inds.sum() - intersection
        union = union.to(torch.float32)
        if union == 0:
            # if there is no ground truth, do not include in evaluation
            ious[cl] = float('nan')  
        else:
            ious[cl] = float(intersection) / max(union, 1)

    return ious.reshape((1, num_classes))

def pixelwise_acc(pred, target):
    pred = pred.float()
    target = target.float()
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def dice_score(preds, targets, logger=None):
    smooth = 5e-3
    num = preds.size(0)              # batch size
    preds_flat = preds.view(num, -1).float()
    targets_flat = targets.view(num, -1).float()
    
    intersection = (preds_flat * targets_flat).sum()

#     if logger:
#         logger.debug('-Dice score- intersection: {:.2f}, preds: {:.2f}, targets: {:.2f}'.format(intersection,\
#             preds_flat.sum(),\
#             targets_flat.sum()))
#     else:
#         print('-Dice score- intersection: {:.2f}, preds: {:.2f}, targets: {:.2f}'.format(intersection,\
#             preds_flat.sum(),\
#             targets_flat.sum()))
    
    return (2. * intersection + smooth)/(preds_flat.sum() + targets_flat.sum() + smooth)

def sen_score(preds, targets, logger=None):
    smooth = 5e-3
    num = preds.size(0)              # batch size
    preds_flat = preds.view(num, -1).float()
    targets_flat = targets.view(num, -1).float()
    intersection = (preds_flat * targets_flat).sum()
    return (intersection + smooth)/(targets_flat.sum() + smooth)
