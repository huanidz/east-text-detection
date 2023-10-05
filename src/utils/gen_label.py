import numpy as np
import cv2
from typing import List
from .math import L2_distance, shrink_point
from .common import scale_contour, sorting_quadrangle_TLTRBRBL, findCoverRectWithMinimalArea

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
    
    Scaled_Quadrangles = []
    for annotation in annotations:
        p1 = [annotation[0]["x"], annotation[0]["y"]]
        p2 = [annotation[1]["x"], annotation[1]["y"]]
        p3 = [annotation[2]["x"], annotation[2]["y"]]
        p4 = [annotation[3]["x"], annotation[3]["y"]]

        # Quadrangle = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}
        Quadrangle = np.array([[p1], [p2], [p3], [p4]])
                
        # To OpenCV contour (prepare for scaling inward)
        Inward_Scaled_Quadrangle = scale_contour(Quadrangle, INWARD_MOVING_RATE)
        Scaled_Quadrangles.append(Inward_Scaled_Quadrangle)
        
        # cv2.drawContours(score_map, [Quadrangle], -1, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.drawContours(score_map, [Inward_Scaled_Quadrangle], -1, (255, 255, 255), -1, cv2.LINE_AA)
    
    return image, score_map, Scaled_Quadrangles

def gen_label(img_path, label_path, target_size):
    image, score_map, scaled_quadrangles = gen_score_map(img_path=img_path, label_path=label_path)    
    
    image = cv2.resize(image, (target_size, target_size))
    score_map = cv2.resize(score_map, (target_size, target_size))
    
    rect_boxes = []
    for quadrangle in scaled_quadrangles:
        rect_box = findCoverRectWithMinimalArea(quadrangle)
        rect_boxes.append(rect_box)
    
    return image, score_map
    
            
       
        