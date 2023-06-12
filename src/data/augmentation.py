from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from typing import Any

import cv2
import numpy as np
import albumentations as A

from .utils import *

class RamdomScale:
    def __init__(self, scale_range, crop_size) -> None:
        self.scale_size = np.random.randint(*scale_range)
        self.crop_size = crop_size

    def __call__(self, image, bboxes, p):
        if np.random.uniform() < p:
            image, bboxes = Resize((self.scale_size, self.scale_size))(image, bboxes)
            image, bboxes = RandomCrop(self.crop_size)(image, bboxes)
        return image, bboxes
    

class Translation:
    def __init__(self, ratio_shift) -> None:
        self.ratio_shift = ratio_shift

    def __call__(self, image, bboxes, p):
        if np.random.uniform() < p:
            bboxes = bboxes.copy()
            
            image = image.copy()
            h, w = image.shape[:2]
            shift_x = np.random.uniform(-self.ratio_shift, self.ratio_shift)
            shift_y = np.random.uniform(-self.ratio_shift, self.ratio_shift)
            trans_matrix = np.array([
                [1, 0, shift_x*w],
                [0, 1, shift_y*h]],
                np.float32
            )
            image = cv2.warpAffine(image, trans_matrix, (w, h), borderValue=(128,128,128))
            bboxes[:, :2] += [shift_x*w, shift_y*h]
            bboxes[:, 2:] += [shift_x*w, shift_y*h]
            bboxes[:, :2] = np.maximum([2, 2], bboxes[:, :2])
            bboxes[:, 2:] = np.minimum([w-2, h-2], bboxes[:, 2:])
        return image, bboxes


class HSV:
    def __init__(self, h_factor, s_factor, v_factor):
        self.h_factor = h_factor
        self.s_factor = s_factor
        self.v_factor = v_factor

    def __call__(self, image, p):
        if np.random.uniform() < p:
            image = image.copy()
            # Convert RGB to HSV
            # Only randomly adjust exposure and saturation
            r = np.random.uniform(-1, 1, 3) * [self.h_factor, self.s_factor, self.v_factor] + 1
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8)
            h, s, v = cv2.split(image_hsv)
            dtype = image.dtype

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image_hsv = cv2.merge((cv2.LUT(h, lut_hue), cv2.LUT(s, lut_sat), cv2.LUT(v, lut_val)))

            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image).astype(np.float32)
        return image
    

class AlbumAug:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.BBoxSafeRandomCrop(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Affine(p=0.3),
            A.ShiftScaleRotate(p=0.2, rotate_limit=15),
            # A.RandomBrightnessContrast(p=0.5),
            #A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2),
        )
    
    def __call__(self, image, bboxes, labels):
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        transformed_image = transformed['image'] 
        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_labels = transformed['labels']
        return transformed_image, transformed_bboxes, transformed_labels
