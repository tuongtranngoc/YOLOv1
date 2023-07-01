from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from typing import Any

import cv2
import numpy as np
import albumentations as A

from .utils import *


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
    

class AlbumAug:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.BBoxSafeRandomCrop(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Affine(p=0.3),
            A.ShiftScaleRotate(p=0.2, rotate_limit=15),
            # A.RandomBrightnessContrast(p=0.3),
            # A.HueSaturationValue()
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
