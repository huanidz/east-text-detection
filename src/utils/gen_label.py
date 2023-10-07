import numpy as np
import cv2
from typing import List
from .common import scale_contour, cal_shortest_edge_length
from time import time
from tqdm import tqdm

# FOR TESTING
IMG_PATH = "../../data/ICDAR_2015/train_images/img_1.jpg"
LABEL_PATH = "../../data/ICDAR_2015/train_label/gt_img_1.txt"

def get_shrink_quad(original_quadrangle):
    
    """
    The input quadrangle must be in clockwise order and from top-left to bottom-right
    """
    
    p1 = np.array(original_quadrangle[0], dtype=np.float32) # Top left
    p2 = np.array(original_quadrangle[1], dtype=np.float32) # Top right
    p3 = np.array(original_quadrangle[2], dtype=np.float32) # Bottom right
    p4 = np.array(original_quadrangle[3], dtype=np.float32) # Bottom left
    
    # Clockwise-L2
    D_p12 = np.linalg.norm([p1, p2])
    D_p23 = np.linalg.norm([p2, p3])
    D_p34 = np.linalg.norm([p3, p4])
    D_p41 = np.linalg.norm([p4, p1])
    
    # Shared vertice with clockwise-L2
    D_p14 = np.linalg.norm([p1, p4])
    D_p21 = np.linalg.norm([p2, p1])
    D_p32 = np.linalg.norm([p3, p2])
    D_p43 = np.linalg.norm([p4, p3])
    
    r1 = min(D_p12, D_p14)
    r2 = min(D_p21, D_p23)
    r3 = min(D_p34, D_p32)
    r4 = min(D_p41, D_p43)
    
    pair_D1234 = np.mean([D_p12, D_p34]) # TODO: can be simplified to just add.
    pair_D2341 = np.mean([D_p23, D_p41])
    
    R = 0.3
    
    if pair_D1234 >= pair_D2341: # Shrink pair 12 and 34 first and then 2341
        # Shrink 12 34
        
        direction = (p2 - p1) # 1
        norm_direction = (direction)/np.linalg.norm(direction)
        p1 = p1 +  norm_direction * (R * r1)
        
        direction = (p1 - p2) # 2
        norm_direction = (direction)/np.linalg.norm(direction)
        p2 = p2 + norm_direction * (R * r2)
        
        direction = (p4 - p3) # 3
        norm_direction = (direction)/np.linalg.norm(direction)
        p3 = p3 +  norm_direction * (R * r3)
        
        direction = (p3 - p4) # 4
        norm_direction = (direction)/np.linalg.norm(direction)
        p4 = p4 + norm_direction * (R * r4)
        
        # Shrink 23 41
    
        direction = (p3 - p2) # 2
        norm_direction = (direction)/np.linalg.norm(direction)
        p2 = p2 +  norm_direction * (R * r2)
        
        direction = (p2 - p3) # 3
        norm_direction = (direction)/np.linalg.norm(direction)
        p3 = p3 + norm_direction * (R * r3)
        
        direction = (p1 - p4) # 4
        norm_direction = (direction)/np.linalg.norm(direction)
        p4 = p4 +  norm_direction * (R * r4)
        
        direction = (p4 - p1) # 1
        norm_direction = (direction)/np.linalg.norm(direction)
        p1 = p1 + norm_direction * (R * r1)
    
    else:
        # Shrink 23 41
    
        direction = (p3 - p2) # 2
        norm_direction = (direction)/np.linalg.norm(direction)
        p2 = p2 +  norm_direction * (R * r2)
        
        direction = (p2 - p3) # 3
        norm_direction = (direction)/np.linalg.norm(direction)
        p3 = p3 + norm_direction * (R * r3)
        
        direction = (p1 - p4) # 4
        norm_direction = (direction)/np.linalg.norm(direction)
        p4 = p4 +  norm_direction * (R * r4)
        
        direction = (p4 - p1) # 1
        norm_direction = (direction)/np.linalg.norm(direction)
        p1 = p1 + norm_direction * (R * r1)

        # Shrink 12 34
        
        direction = (p2 - p1) # 1
        norm_direction = (direction)/np.linalg.norm(direction)
        p1 = p1 +  norm_direction * (R * r1)
        
        direction = (p1 - p2) # 2
        norm_direction = (direction)/np.linalg.norm(direction)
        p2 = p2 + norm_direction * (R * r2)
        
        direction = (p4 - p3) # 3
        norm_direction = (direction)/np.linalg.norm(direction)
        p3 = p3 +  norm_direction * (R * r3)
        
        direction = (p3 - p4) # 4
        norm_direction = (direction)/np.linalg.norm(direction)
        p4 = p4 + norm_direction * (R * r4)
        
    shrink_quad = np.array([np.int32(p1), np.int32(p2), np.int32(p3), np.int32(p4)])
    return shrink_quad    

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
        # Inward_Scaled_Quadrangle = scale_contour(Quadrangle, INWARD_MOVING_RATE)
        shrink_quad = get_shrink_quad(Quadrangle)
        
        cv2.drawContours(score_map, [shrink_quad], -1, 1.0, -1)
    

    
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
    
            
def gen_label_V2(img_path, label_path, target_size):
    
    """
    Input:
    1. Original image
    2. Annotations
    3. Target preprocess size
    
    Generate:
    1. Score map
    2. Geo map
    """
    
    image = cv2.imread(filename=img_path, flags=cv2.IMREAD_COLOR)
    
    H, W, C = image.shape
    Net_out_H = target_size // 4
    Net_out_W = target_size // 4
    
    # Scale height and width ratios to be match with network's output
    Scaled_H_Ratio = Net_out_H / H
    Scaled_W_Ratio = Net_out_W / W

    # Normalizing
    image = cv2.resize(image, (target_size, target_size))
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    image = image / 255.0
    image = (image - img_mean) / img_std # Image is now ready to be input with size of Target_Size
    
    # Re-annotate
    annotations = get_annotation_coordinates(label_path)
    quadrangles = []
    for annotation in annotations:
        annotation[0]["x"], annotation[0]["y"] = annotation[0]["x"] * Scaled_W_Ratio, annotation[0]["y"] * Scaled_H_Ratio
        annotation[1]["x"], annotation[1]["y"] = annotation[1]["x"] * Scaled_W_Ratio, annotation[1]["y"] * Scaled_H_Ratio
        annotation[2]["x"], annotation[2]["y"] = annotation[2]["x"] * Scaled_W_Ratio, annotation[2]["y"] * Scaled_H_Ratio
        annotation[3]["x"], annotation[3]["y"] = annotation[3]["x"] * Scaled_W_Ratio, annotation[3]["y"] * Scaled_H_Ratio
        
        p1 = [annotation[0]["x"], annotation[0]["y"]]
        p2 = [annotation[1]["x"], annotation[1]["y"]]
        p3 = [annotation[2]["x"], annotation[2]["y"]]
        p4 = [annotation[3]["x"], annotation[3]["y"]]
        
        quadrangles.append([p1, p2, p3, p4])
        
    shrunk_quadrangles = []
    for quadrangle in quadrangles:
        s_quadrangle = get_shrink_quad(quadrangle)
        shrunk_quadrangles.append(s_quadrangle)
        
    # Generate score map
    score_map = np.zeros((Net_out_H, Net_out_W), dtype=np.uint8)
    for quad in shrunk_quadrangles:
        quad = quad.reshape((-1,1,2))
        cv2.fillPoly(score_map, [quad], (255))
    score_map = np.where(score_map == 255, 1, 0).astype(np.float32)

    # Generate geo map
    geo_map = np.zeros((8, Net_out_H, Net_out_W), dtype=np.float32)    
    for quad in shrunk_quadrangles:
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(quad)

        # Ensure the bounding rectangle is within the frame
        rect_x = max(0, rect_x)
        rect_y = max(0, rect_y)
        rect_x = min(rect_x, Net_out_W)
        rect_y = min(rect_y, Net_out_H)

        # Adjust width & height to avoid going out of bounds
        rect_w = min(rect_w, Net_out_W - rect_x)
        rect_h = min(rect_h, Net_out_H - rect_y)

        for i, corner_pts in enumerate(quad):      
            
            for y in range(rect_y, rect_y + rect_h):
                for x in range(rect_x, rect_x + rect_w):
                    if cv2.pointPolygonTest(quad, (x, y), False) >= 0:
                        geo_map[2 * i, y, x] = corner_pts[0] - x
                        geo_map[2 * i + 1, y, x] = corner_pts[1] - y
    
    return image, score_map, geo_map                    
    
        
        
    
    
        