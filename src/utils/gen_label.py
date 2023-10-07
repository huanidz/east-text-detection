import numpy as np
import cv2
from typing import List
from .common import scale_contour, cal_shortest_edge_length
from time import time
from tqdm import tqdm

# FOR TESTING
IMG_PATH = "../../data/ICDAR_2015/train_images/img_1.jpg"
LABEL_PATH = "../../data/ICDAR_2015/train_label/gt_img_1.txt"

def get_annotation_coordinates(label_path) -> List:
    
    """
    FORMAT: [[{'x': '377', 'y': '117'}, {'x': '463', 'y': '117'}, {'x': '465', 'y': '130'}, {'x': '378', 'y': '130'}], ...]
    """
        
    with open(label_path, 'r') as file:
        annotations = file.readlines()
    
    # Cleaning
    annotations = [x.replace("\n", "").replace("\ufeff", "") for x in annotations]
    
    for i in range(len(annotations)):
        for j in range(8):
            annotations[i] = annotations[i].replace(",", "<SEP>", 1)
            
    annotations = [x.split("<SEP>")[:-1] for x in annotations]
    
    annotations_dict_style = []
    for annotation in annotations:
        one_annotation = []
        for i in range(0, len(annotation), 2):
            x = int(annotation[i])
            y = int(annotation[i + 1])
            one_annotation.append({"x":x, "y":y})
        annotations_dict_style.append(one_annotation)
    
    
    return annotations_dict_style
    

#ICDAR_2015
def gen_score_map(img_path, label_path):
    
    """
    Q = {p1, p2, p3, p4}:
    p1 - Top Left: x1, y1
    p2 - Top Right: x2, y2
    p3 - Bottom Right: x3, y3
    p4 - Bottom Left: x4, y4
    
    CONST: INWARD_MOVING_RATE
    """
    INWARD_MOVING_RATE = 0.7 # 1 - 0.3
    
    image = cv2.imread(filename=img_path)
    score_map = np.zeros(image.shape[:2], dtype=np.uint8)
    
    annotations = get_annotation_coordinates(label_path)
    
    # Edge case #1: No annotation in image
    if len(annotations) == 0:
        return np.zeros_like(image)
    
    for annotation in annotations:
        p1 = [annotation[0]["x"], annotation[0]["y"]]
        p2 = [annotation[1]["x"], annotation[1]["y"]]
        p3 = [annotation[2]["x"], annotation[2]["y"]]
        p4 = [annotation[3]["x"], annotation[3]["y"]]

        # Quadrangle = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}
        Quadrangle = np.array([[p1], [p2], [p3], [p4]])
                
        # To OpenCV contour (prepare for scaling inward)
        Inward_Scaled_Quadrangle = scale_contour(Quadrangle, INWARD_MOVING_RATE)
        
        cv2.drawContours(score_map, [Inward_Scaled_Quadrangle], -1, 1.0, -1)
    
    num_ones = np.count_nonzero(score_map == 1.0)

    print("first:", num_ones)
    
    return image, score_map, annotations

def gen_label(img_path, label_path, target_size):
    image, score_map, annotations = gen_score_map(img_path=img_path, label_path=label_path)    
    H, W, C = image.shape
    
    scale_x = 512.0 / W
    scale_y = 512.0 / H
    final_scale_x = scale_x * (128.0 / 512.0)
    final_scale_y = scale_y * (128.0 / 512.0)
    
    # Input image
    image = cv2.resize(image, (target_size, target_size))
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    image = image / 255.0
    image = (image - img_mean) / img_std
    num_ones = np.count_nonzero(score_map == 1.0)

    print("second:", num_ones)
    score_map = cv2.resize(score_map, (128, 128)) # Mask !NOTE: resize cause value changed due to interpolation step
    if (score_map.min() == 0.0) and (score_map.max() == 1.0):
        print("All values are 0.0 or 1.0")
    else:
        print("Has other values")
        
    num_ones = np.count_nonzero(score_map == 1.0)

    print("second:", num_ones)

    
    geo_map = np.zeros((8, 128, 128), dtype=np.float32)
    
    contours = []
    for annotation in annotations:
        p1 = [int(annotation[0]["x"] * final_scale_x), int(annotation[0]["y"] * final_scale_y)]
        p2 = [int(annotation[1]["x"] * final_scale_x), int(annotation[1]["y"] * final_scale_y)]
        p3 = [int(annotation[2]["x"] * final_scale_x), int(annotation[2]["y"] * final_scale_y)]
        p4 = [int(annotation[3]["x"] * final_scale_x), int(annotation[3]["y"] * final_scale_y)]
        contours.append(np.array([[p1], [p2], [p3], [p4]]))
    
    #Loop through each contour
    for contour in contours:
        
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
        
        # calculating the N_Q_Star
        # shortest_edge_length = cal_shortest_edge_length(contour)
        
        # Building the geo_map
        for i, corner_pts in enumerate(contour):
            
            for y in range(rect_y, rect_y + rect_h):
                for x in range(rect_x, rect_x + rect_w):

                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        geo_map[2 * i, y, x] = corner_pts[0][0] - x
                        geo_map[2 * i + 1, y, x] = corner_pts[0][1] - y
                        
    
    return image, score_map, geo_map
    
            
       
        