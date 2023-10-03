# Package
import cv2
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from glob import glob
from natsort import natsorted
from tqdm import tqdm

# Project Source Package
from src.components.backbones.ResNet18 import ResNet18
from src.components.necks.EASTFPN import EASTFPN
from src.components.heads.EASTHead import EASTHead
from src.components.losses.EASTLoss import EASTLoss
from src.utils.common import count_parameters
from src.utils.gen_label import gen_score_map, gen_image_label_pair


"""TESTING MODELS"""
if __name__ == "__main__":
    input = torch.randn((1, 3, 512, 512)).float()
    random_target = torch.randn((1, 1, 128,128)).float()

    backbone = ResNet18()
    output = backbone(input)
    
    neck = EASTFPN(output.shape[1], 64)
    output = neck(output)
    
    head = EASTHead(neck_out_channels=64)
    output = head(output)
    y_score = output["y_score"]
    print(f"==>> y_score.shape: {y_score.shape}")
    y_geo = output["y_geo"]
    
    mask = torch.randint(low=0, high=2, size=(1, 1, 128, 128)).float()
    
    # ScoreMapPrediction, ScoreMapGT, GeoPrediction, GeoGT
    random_target = torch.randn((1, 1, 128, 128))
    # random_target_2 = torch.randn((1, 128, 128)).float()
    CE = nn.BCEWithLogitsLoss()
    loss = CE(y_score, mask)
    
    
    loss_func = EASTLoss()
    loss = loss_func(torch.sigmoid(y_score), mask)
    print(loss.item())
        
        
    
"""TESTING LABEL GENERATION"""
# IMG_PATH = "/home/huan/prjdir/east-text-detection/data/ICDAR_2015/train_images/img_1.jpg"
# LABEL_PATH = "/home/huan/prjdir/east-text-detection/data/ICDAR_2015/train_label/gt_img_1.txt"

# if __name__ == "__main__":
#     score_map = gen_score_map(img_path=IMG_PATH, label_path=LABEL_PATH)
    

"""TESTING DATASET PREPROCESSING"""
# IMAGE_PATH = "/home/huan/prjdir/east-text-detection/data/ICDAR_2015/train_images"
# LABEL_PATH = "/home/huan/prjdir/east-text-detection/data/ICDAR_2015/train_label"
# TARGET_SIZE = 512

# PREPROCESSED_DIRECTORY = "data/preprocess/"
# PREPROCESSED_DIRECTORY_TRAIN_IMAGES = f"{PREPROCESSED_DIRECTORY}/train/images/"
# PREPROCESSED_DIRECTORY_TRAIN_LABEL = f"{PREPROCESSED_DIRECTORY}/train/label/"

# if __name__ == "__main__":
    
    
#     os.makedirs(PREPROCESSED_DIRECTORY, exist_ok=True)
#     os.makedirs(PREPROCESSED_DIRECTORY_TRAIN_IMAGES, exist_ok=True)
#     os.makedirs(PREPROCESSED_DIRECTORY_TRAIN_LABEL, exist_ok=True)
#     print("Directories has been created!")
    
#     image_paths = natsorted(glob(f"{IMAGE_PATH}/*"))
#     label_paths = natsorted(glob(f"{LABEL_PATH}/*"))
    
#     for image_path, label_path in zip(image_paths, label_paths):
#         print(f"Processing image: {image_path}, label: {label_path}")
#         image, label_mask = gen_image_label_pair(img_path=image_path, label_path=label_path, target_size=TARGET_SIZE)        
#         cv2.imwrite(f"{PREPROCESSED_DIRECTORY_TRAIN_IMAGES}/{os.path.basename(image_path)}", image)
#         cv2.imwrite(f"{PREPROCESSED_DIRECTORY_TRAIN_LABEL}/{os.path.basename(image_path)}", label_mask)
    
