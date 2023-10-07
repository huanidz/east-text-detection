import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = epsilon
        
    def forward(self, pred, gt):
        
        # This is used when pred is output logits
        pred = torch.sigmoid(pred)
        
        intersection = torch.sum(pred * gt) # pred should be sigmoied.
        union = torch.sum(pred * gt) + torch.sum(gt) + self.eps
        loss = 1 - (2 * intersection / union)
        return loss
    
    
class EASTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()    
        self.channels = 8
    
    def forward(self, score_map_pred, score_map_gt, geo_pred, geo_gt):
        
        # Dice loss
        dice_loss = self.dice_loss(score_map_pred, score_map_gt)
        
        pred_geo_split = torch.split(geo_pred, split_size_or_sections=8, dim=1)[0]
        label_geo_split = torch.split(geo_gt, split_size_or_sections=8, dim=1)[0]
        
        smooth_l1_loss = 0.0
        for i in range(self.channels):
            abs_different = torch.abs(label_geo_split[:, i] - pred_geo_split[:, i])
            pair_loss = torch.where(abs_different < 1.0, 0.5 * abs_different ** 2, abs_different - 0.5) / 8 
            smooth_l1_loss += pair_loss
        smooth_l1_loss = torch.mean(smooth_l1_loss * score_map_gt)    
        final_loss =  0.01* dice_loss + smooth_l1_loss
        return final_loss