from itertools import tee
import cv2
import numpy as np
import math

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def count_parameters(model, count_non_trainable=False):
    if count_non_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def clockwise_vertices_sorting(vertices_coordinates):
    pass

def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def sorting_quadrangle_TLTRBRBL(quadrangle):
    
    sorted_vertices = sorted(quadrangle, key=lambda vertex: vertex[0])
    left_most_vertices_X = sorted_vertices[:2]
    
    sorted_vertices = sorted(quadrangle, key=lambda vertex: vertex[0], reverse=True)
    right_most_vertices_X = sorted_vertices[:2]
    
    left_points_sorted_XY = sorted(left_most_vertices_X, key=lambda point: point[1])
    right_points_sorted_XY = sorted(right_most_vertices_X, key=lambda point: point[1])
    
    TopLeft, BottomLeft = left_points_sorted_XY[0], left_points_sorted_XY[1]
    TopRight, BottomRight = right_points_sorted_XY[0], right_points_sorted_XY[1]
    
    return TopLeft, TopRight, BottomRight, BottomLeft
