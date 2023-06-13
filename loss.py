import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

binary = True


class Binary_ICNetLoss(nn.BCEWithLogitsLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self,**kwargs):
        super(Binary_ICNetLoss, self).__init__()

        
        

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        _, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
       
        target = target.unsqueeze(1).float()

        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1)
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1)
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1)
        


        loss1 = super(Binary_ICNetLoss, self).forward(pred_sub4.squeeze_(1) , target_sub4)
        loss2 = super(Binary_ICNetLoss, self).forward(pred_sub8.squeeze_(1) , target_sub8)
        loss3 = super(Binary_ICNetLoss, self).forward(pred_sub16.squeeze_(1) , target_sub16)
        
        return loss1 + loss2  + loss3 

class Multiclass_ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self,**kwargs):
        super(Multiclass_ICNetLoss, self).__init__(ignore_index=-1)
        
  

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        _, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
       
        target = target.unsqueeze(1).float()

        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        
        

        loss1 = super(Multiclass_ICNetLoss, self).forward(pred_sub4 , target_sub4)
        loss2 = super(Multiclass_ICNetLoss, self).forward(pred_sub8 , target_sub8)
        loss3 = super(Multiclass_ICNetLoss, self).forward(pred_sub16 , target_sub16)
        
        return loss1 + loss2  + loss3 

