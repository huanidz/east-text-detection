import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from ...utils.math import L2_distance


class ClassBalanceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, Y_prediction, Y_groundtruth):
        beta = 1 - torch.sum(Y_prediction) / (128**2) # Modify the 128 as the neck's output size
        loss = -beta * Y_groundtruth * torch.log(Y_prediction) - (1 - beta)*(1 - Y_groundtruth) * torch.log(1 - Y_prediction)
        print(f"==>> loss: {loss}")
        return loss.mean()

class NormalizedSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, Q_prediction, Q_groundtruth):
        N_Q_gt_star = torch.abs(torch.sum(L2_distance((Q_groundtruth[:, 0], Q_groundtruth[:, 1]),
                                  (Q_groundtruth[:, 2], Q_groundtruth[:, 3]))))
        
        for i in range(4):
            if i < 4 - 1:
                p_i_x = Q_groundtruth[:, 2*i]
                p_i_y = Q_groundtruth[:, 2*i + 1]
                p_imod4add1_x = Q_groundtruth[:, 2*i + 2]
                p_imod4add1_y = Q_groundtruth[:, 2*i + 3]
            elif i == (4 - 1):
                p_i_x = Q_groundtruth[:, 2*i]
                p_i_y = Q_groundtruth[:, 2*i + 1]
                p_imod4add1_x = Q_groundtruth[:, 0]
                p_imod4add1_y = Q_groundtruth[:, 1]
                
            D_L2 = torch.abs(torch.sum(L2_distance(point_1=(p_i_x, p_i_y), 
                                                   point_2=(p_imod4add1_x, p_imod4add1_y))))
            N_Q_gt_star = min(N_Q_gt_star, D_L2)
        
        abs_diff = torch.abs(Q_prediction - Q_groundtruth)
        smooth_l1 = torch.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)
        loss = smooth_l1.sum() / (8 * N_Q_gt_star)
        return loss

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
    
# class EastSmoothL1Loss(nn.Module):
#     def __init__(self):
#         super(EastSmoothL1Loss, self).__init__()
#         self.channels = 8
        
#     def forward(self, geo_predict, geo_targets):
#         pred_geo_split = torch.split(geo_predict, split_size_or_sections=8, dim=1) # Splitting channels of geo_map (predict)
#         label_geo_split = torch.split(geo_targets, split_size_or_sections=8, dim=1) # Splitting channels of geo_map (label)
        
#         smooth_l1 = 0.
#         for i in range(self.channels):
#             geo_different = torch.abs(label_geo_split[i] - pred_geo_split[i])
#             smooth_l1_sign = torch.less(geo_different, )

# class EASTLoss(nn.Module):
#     def __init__(self, L_ScoreMap_Weight: float = 1.0, L_Geo_Weight: float = 1.0):
#         super().__init__()
#         self.LossScoreMap = ClassBalanceCrossEntropyLoss()
#         self.LossGeo = NormalizedSmoothL1Loss()
#         self.L_ScoreMap_Weight = L_ScoreMap_Weight
#         self.L_Geo_Weight = L_Geo_Weight    
    
#     def forward(self, ScoreMapPrediction, ScoreMapGT, GeoPrediction, GeoGT):
#         LossScoreMap = self.LossScoreMap(ScoreMapPrediction, ScoreMapGT)
#         LossGeo = self.LossGeo(GeoPrediction, GeoGT)
#         final_loss = self.L_ScoreMap_Weight * LossScoreMap + self.L_Geo_Weight * LossGeo
#         return final_loss       
    
class EASTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()    
        self.channels = 8
    
    def forward(self, score_map_pred, score_map_gt, geo_pred, geo_gt, n_q_star_map):
        
        mask = torch.where(score_map_gt == 255.0, 1.0, 0.0)
        
        pred_geo_split = torch.split(geo_pred, split_size_or_sections=8, dim=1)[0]
        label_geo_split = torch.split(geo_gt, split_size_or_sections=8, dim=1)[0]
        
        smooth_l1_loss = 0.0
        for i in range(self.channels):
            abs_different = torch.abs(label_geo_split[i] - pred_geo_split[i])
            pair_loss = torch.where(abs_different < 1.0, 0.5 * abs_different ** 2, abs_different - 0.5) / 8 * n_q_star_map * mask
            smooth_l1_loss += pair_loss
        
        smooth_l1_loss = torch.mean(smooth_l1_loss * mask)    
        dice_loss = self.dice_loss(score_map_pred, mask)
        final_loss = 0.1*dice_loss + smooth_l1_loss
        return final_loss