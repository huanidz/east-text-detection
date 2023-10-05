# Package
import cv2
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm


# Project Source Package
from src.components.backbones.ResNet18 import ResNet18
from src.components.necks.EASTFPN import EASTFPN
from src.components.heads.EASTHead import EASTHead
from src.components.losses.EASTLoss import EASTLoss
from src.utils.common import count_parameters, findCoverRectWithMinimalArea
from src.utils.gen_label import gen_score_map, gen_label
from src.components.net.EAST import EASTNet

"""TESTING MODELS"""
# if __name__ == "__main__":
#     input = torch.randn((1, 3, 512, 512)).float()
#     random_target = torch.randn((1, 1, 128,128)).float()

#     backbone = ResNet18()
#     output = backbone(input)
    
#     neck = EASTFPN(output.shape[1], 64)
#     output = neck(output)
    
#     head = EASTHead(neck_out_channels=64)
#     output = head(output)
#     y_score = output["y_score"]
#     print(f"==>> y_score.shape: {y_score.shape}")
#     y_geo = output["y_geo"]
    
#     mask = torch.randint(low=0, high=2, size=(1, 1, 128, 128)).float()
    
#     loss_func = EASTLoss()
#     loss = loss_func(torch.sigmoid(y_score), mask)
#     print(loss.item())
        
        
    
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
# PREPROCESSED_DIRECTORY_TRAIN_MASK = f"{PREPROCESSED_DIRECTORY}/train/mask/"

# if __name__ == "__main__":
    
    
#     os.makedirs(PREPROCESSED_DIRECTORY, exist_ok=True)
#     os.makedirs(PREPROCESSED_DIRECTORY_TRAIN_IMAGES, exist_ok=True)
#     os.makedirs(PREPROCESSED_DIRECTORY_TRAIN_LABEL, exist_ok=True)
#     print("Directories has been created!")
    
#     image_paths = natsorted(glob(f"{IMAGE_PATH}/*"))
#     label_paths = natsorted(glob(f"{LABEL_PATH}/*"))
    
#     for image_path, label_path in zip(image_paths, label_paths):
#         print(f"Processing image: {image_path}, label: {label_path}")
#         image, label_mask = gen_label(img_path=image_path, label_path=label_path, target_size=TARGET_SIZE)        
#         cv2.imwrite(f"{PREPROCESSED_DIRECTORY_TRAIN_IMAGES}/{os.path.basename(image_path)}", image)
#         cv2.imwrite(f"{PREPROCESSED_DIRECTORY_TRAIN_LABEL}/{os.path.basename(image_path)}", label_mask)


"""
TESTING GEO MAP LABEL GENERATION
"""
# IMAGE_PATH = "/home/huan/prjdir/east-text-detection/data/text-detection/train/training_images"
# LABEL_PATH = "/home/huan/prjdir/east-text-detection/data/text-detection/train/label"

# # IMAGE_PATH = "/home/huan/prjdir/east-text-detection/data/text-detection/train/train_1"
# # LABEL_PATH = "/home/huan/prjdir/east-text-detection/data/text-detection/train/label_1"

# TARGET_SIZE = 512

# PREPROCESSED_DIRECTORY = "data/preprocess"
# PREPROCESSED_DIRECTORY_TRAIN_IMAGES = f"{PREPROCESSED_DIRECTORY}/train/images"
# PREPROCESSED_DIRECTORY_TRAIN_LABEL = f"{PREPROCESSED_DIRECTORY}/train/label"
# if __name__ == "__main__":
    
#     os.makedirs(PREPROCESSED_DIRECTORY, exist_ok=True)
#     os.makedirs(PREPROCESSED_DIRECTORY_TRAIN_IMAGES, exist_ok=True)
#     os.makedirs(PREPROCESSED_DIRECTORY_TRAIN_LABEL, exist_ok=True)
#     print("Directories has been created!")
    
#     image_paths = natsorted(glob(f"{IMAGE_PATH}/*"))
#     label_paths = natsorted(glob(f"{LABEL_PATH}/*"))
    
#     for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
#         print(f"Processing image: {image_path}, label: {label_path}")
#         image, label_mask, geo_map = gen_label(img_path=image_path, label_path=label_path, target_size=TARGET_SIZE)        
#         cv2.imwrite(f"{PREPROCESSED_DIRECTORY_TRAIN_IMAGES}/{os.path.basename(image_path)}", image)
#         np.save(f"{PREPROCESSED_DIRECTORY_TRAIN_LABEL}/{os.path.splitext(os.path.basename(image_path))[0]}.npy", 
#                 np.concatenate([geo_map, label_mask[np.newaxis, :]], axis=0))


"""
TEST FORWARD AND BACKWARD FULL
"""
IMAGE_PATH = "/home/huan/prjdir/east-text-detection/data/text-detection/train/train_1"
LABEL_PATH = "/home/huan/prjdir/east-text-detection/data/text-detection/train/label_1"

TARGET_SIZE = 512

PREPROCESSED_DIRECTORY = "data/preprocess"
PREPROCESSED_DIRECTORY_TRAIN_IMAGES = f"{PREPROCESSED_DIRECTORY}/train/images"
PREPROCESSED_DIRECTORY_TRAIN_LABEL = f"{PREPROCESSED_DIRECTORY}/train/label"

if __name__ == "__main__":
    image_paths = natsorted(glob(f"{IMAGE_PATH}/*"))
    label_paths = natsorted(glob(f"{LABEL_PATH}/*"))
    
    for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        print(f"Processing image: {image_path}, label: {label_path}")
        image, label_mask, geo_map = gen_label(img_path=image_path, label_path=label_path, target_size=TARGET_SIZE)     

    torch.cuda.empty_cache()
    input = torch.from_numpy(image[np.newaxis, :]).permute(0, 3, 1, 2).float().to("cuda")
    score_target = torch.from_numpy(label_mask[np.newaxis, np.newaxis, :]).float().to("cuda")
    geo_target = torch.from_numpy(geo_map[np.newaxis, :]).float().to("cuda")
    
    # output = head(output)
    model = EASTNet().to("cuda")
    print("Net params:", count_parameters(model))
    

    output = model(input)
    y_score = output["y_score"]
    y_geo = output["y_geo"]
    
    loss_func = EASTLoss().to("cuda")
    loss = loss_func(torch.sigmoid(y_score), score_target, y_geo, geo_target)
    print(loss.item())
    loss.backward()
    
    cv2.imshow("ASD", image)
    cv2.waitKey(0)
    
