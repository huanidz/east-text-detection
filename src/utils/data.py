import torch
from torch.utils.data import Dataset
from natsort import natsorted
from glob import glob
from .gen_label import gen_label
import numpy as np

class EastDataset(Dataset):
    def __init__(self, images_folder, annotations_folder, target_size:int = 512, is_cuda:bool = True) -> None:
        super(EastDataset, self).__init__()
        print(f"==>> images_folder: {images_folder}")
        self.image_paths = natsorted(glob(f"{images_folder}/*"))
        self.label_paths = natsorted(glob(f"{annotations_folder}/*"))
        self.target_size = target_size
        self.is_cuda = is_cuda
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # print(f"Current index: {self.image_paths[idx]}")
        image, score_map, geo_map = gen_label(img_path=self.image_paths[idx],
                                               label_path=self.label_paths[idx],
                                               target_size=self.target_size)
        if self.is_cuda:
            image = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32).cuda()
            score_map = torch.from_numpy(score_map[np.newaxis, :]).to(dtype=torch.float32).cuda()
            geo_map = torch.from_numpy(geo_map).to(dtype=torch.float32).cuda()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32)
            score_map = torch.from_numpy(score_map[np.newaxis, :]).to(dtype=torch.float32)
            geo_map = torch.from_numpy(geo_map).to(dtype=torch.float32)
        
        return image, score_map, geo_map