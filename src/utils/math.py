import numpy as np
import torch

def L2_distance(point_1, point_2):
    """
    Function to calculate L2 distance
    :param p: numpy array of point p
    :param q: numpy array of point q
    """
    return torch.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)

# ICDAR_2015
def shrink_point(p1, p2, r):
    """
    Shrink edge by moving its endpoints inward by ratio of r
    :param p1, p2: tuple of x, y for two ends of an edge
    :param r: reference length ratio
    """
    return p1[0] + (p2[0] - p1[0]) * r, p1[1] + (p2[1] - p1[1]) * r
    