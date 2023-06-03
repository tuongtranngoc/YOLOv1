import os
import cv2
import json
import glob
from tqdm import tqdm

from .utils import *
from ..data import CFG
from .augmentation import *
from .dataset import BaseDatset

import torch
import torch.nn.functional as F


class YoloDatset(BaseDatset):
    def __init__(self, image_path, label_path, txt_path, augment=False) -> None:
        self.cfg = CFG
        self.image_path = image_path    
        self.label_path = label_path
        self.txt_path = txt_path
        self.augment = augment
        self.image_size = self.cfg['image_size']
        self.dataset_voc = self.load_dataset_voc_format(self.image_path, self.label_path, self.txt_path)

    def agumentation(self, image, bboxes):
        image, bboxes = RamdomScale(scale_range=self.cfg['scale_range'], crop_size=self.cfg['crop_size'])(image, bboxes, self.cfg['p_random_scale'])
        image, bboxes = Translation(ratio_shift=self.cfg['ratio_shift'])(image, bboxes, self.cfg['p_translate'])
        image = HSV(self.cfg['h_factor'], self.cfg['s_factor'], self.cfg['v_factor'])(image, self.cfg['p_hsv'])
        return image, bboxes

    def get_image_label(self, image_pth, bboxes):
        image = cv2.imread(image_pth).astype(np.float32)
        if self.augment:
            image, bboxes = self.agumentation(image, bboxes)

        image, bboxes = Resize(self.image_size)(image, bboxes)
        normalized_image = Normalize()(image)
        image_rgb = normalized_image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image_rgb)
        return image_tensor, bboxes
    
    def make_grid_cells(self, cls_ids, boxes):
        S = self.cfg['S'] 
        B = self.cfg['B']
        C = self.cfg['C']
        # Divide the input image into SXS grid cells
        grid_cell_i = self.image_size[0] / S
        grid_cell_j = self.image_size[1] / S
        # Define input shape
        target_cxcywh = torch.zeros((S, S, 5*B + C), dtype=torch.float32)
        
        for class_id, bbox in zip(cls_ids, boxes):
            x_min, y_min, x_max, y_max = bbox.copy()
            # Compute center of an object
            x_c = (x_min + x_max) / 2
            y_c = (y_min + y_max) / 2
            
            # Determine cell position of object
            pos_i = int(x_c // grid_cell_i)
            pos_j = int(y_c // grid_cell_j)

            if target_cxcywh[pos_i, pos_j, 9] == 1.0:
                continue
            # Each grid cell contains:
            #   bbox1: [x_c, y_c, w_box, h_box, p_c, p0, p1, ..., pn]
            #   bbox2: [x_c, y_c, w_box, h_box, p_c, p0, p1, ..., pn]

            p_cls = torch.zeros(C)
            p_cls[int(class_id)-1] = 1.0
            conf_cls = torch.ones((2, )).long()
            box = torch.FloatTensor([
                (x_c - (pos_i * grid_cell_i)) / grid_cell_i,
                (y_c - (pos_j * grid_cell_j)) / grid_cell_j, 
                np.sqrt((x_max - x_min) / self.image_size[0]), 
                np.sqrt((y_max - y_min) / self.image_size[1])]).repeat(B)
                
            # Assign bboxes to each grid cell
            grid_cell = torch.cat([box, conf_cls, p_cls], dim=-1)

            target_cxcywh[pos_i, pos_j, :] = grid_cell
        
        return target_cxcywh

    def __getitem__(self, index):
        image_path, labels = self.dataset_voc[index]
        cls_ids, bboxes = labels[:, 0], labels[:, 1:]
        image, bboxes = self.get_image_label(image_path, bboxes)
        target = self.make_grid_cells(cls_ids, bboxes)
        return image, target

    def __len__(self):
        return len(self.dataset_voc)
