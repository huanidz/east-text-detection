import numpy as np
import torch
import torch.nn as nn
from time import perf_counter
from ...utils.math import L2_distance


class ClassBalanceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, Y_prediction, Y_groundtruth):
        beta = 1 - torch.sum(Y_prediction) / (128**2) # Modify the 128 as the neck's output size
        loss = -beta * Y_groundtruth * torch.log(Y_prediction) - (1 - beta)*(1 - Y_groundtruth) * torch.log(1 - Y_prediction)
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

class EASTLoss(nn.Module):
    def __init__(self, L_ScoreMap_Weight: float = 1.0, L_Geo_Weight: float = 1.0):
        super().__init__()
        self.LossScoreMap = ClassBalanceCrossEntropyLoss()
        self.LossGeo = NormalizedSmoothL1Loss()
        self.L_ScoreMap_Weight = L_ScoreMap_Weight
        self.L_Geo_Weight = L_Geo_Weight    
    
    def forward(self, ScoreMapPrediction, ScoreMapGT, GeoPrediction, GeoGT):
        LossScoreMap = self.LossScoreMap(ScoreMapPrediction, ScoreMapGT)
        LossGeo = self.LossGeo(GeoPrediction, GeoGT)
        final_loss = self.L_ScoreMap_Weight * LossScoreMap + self.L_Geo_Weight * LossGeo
        return final_loss        