import os
import json
import glob

import cv2
import numpy as np
from tqdm import tqdm


def xywh2yolo(bbox, w, h):
    bbox = bbox.copy()
    center_x = (bbox[0] + bbox[2]/2.)/w
    center_y = (bbox[1] + bbox[3]/2.)/h
    width = bbox[2]/w
    height = bbox[3]/h

    return [center_x, center_y, width, height]


def xywh2xyxy(labels):
    # labels = [x1, y1, x2, y2]
    labels = labels.copy()
    labels[:, 2] += labels[:, 0]
    labels[:, 3] += labels[:, 1]

    return labels


def check_bboxes(bb1, bb2):
    bb1 = bb1.copy()
    bb2 = bb2.copy()
    if len(bb2) > 0:
        w1 = bb1[..., 2] - bb1[..., 0]
        h1 = bb1[..., 3] - bb1[..., 1]

        w2 = bb2[..., 2] - bb2[..., 0]
        h2 = bb2[..., 3] - bb2[..., 1]

        idxs_w = np.where(w2 > 0.1*w1)[0]
        idxs_h = np.where(h2 > 0.1*h1)[0]
        idxs = list(set(idxs_h)&set(idxs_w))
        return bb2[idxs]
    return bb2
        

class Resize:
    def __init__(self, image_size) -> None:
        self.image_size = image_size

    def __call__(self, image, bboxes):
        bboxes = bboxes.copy()
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.image_size[0], self.image_size[1]))
        x_scale = self.image_size[0] / w
        y_scale = self.image_size[1] / h
        bboxes[:, 0] *= x_scale
        bboxes[:, 1] *= y_scale
        bboxes[:, 2] *= x_scale
        bboxes[:, 3] *= y_scale
        
        return image, bboxes
    

class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def __call__(self, image, bboxes):
        image = image.copy()
        h, w = image.shape[:2]
        top = np.random.randint(0, w - self.crop_size)
        left = np.random.randint(0, h - self.crop_size)
        bottom, right = top + self.crop_size, left + self.crop_size
        image = image[top:bottom, left:right, :]

        bboxes = bboxes.copy()
        bboxes[:, 0] = np.maximum(left, bboxes[:, 0]) - left
        bboxes[:, 1] = np.maximum(top, bboxes[:, 1]) - top
        bboxes[:, 2] = np.minimum(right, bboxes[:, 2]) - left
        bboxes[:, 3] = np.minimum(bottom, bboxes[:, 3]) - top
        bboxes = check_bboxes(bboxes)

        return image, bboxes


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image):
        image /= 255.
        image -= self.mean
        image /= self.std
        return image


class Unnormalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image):
        image *= self.std
        image += self.mean
        image *= 255.
        return image