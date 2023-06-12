
import cv2
import torch
import torchvision
import numpy as np

from ..config import CFG as cfg
from ..data.utils import Unnormalize


def compute_iou(target, pred, device):
    eps = 1e-6
    x = BoxUtils.decode_yolo(target[..., :4], device)
    y = BoxUtils.decode_yolo(pred[..., :4], device)
    x1 = torch.max(x[..., 0], y[..., 0])
    y1 = torch.max(x[..., 1], y[..., 1])
    x2 = torch.min(x[..., 2], y[..., 2])
    y2 = torch.min(x[..., 3], y[..., 3])
    intersects = torch.clamp((x2-x1), 0) * torch.clamp((y2-y1), 0)
    unions = abs((x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])) + abs((y[..., 2] - y[..., 0]) * (y[..., 3] - y[..., 1])) - intersects
    ious = intersects / (unions + eps)
    return ious

class BoxUtils:
    S = cfg['S']
    B = cfg['B']
    C = cfg['C'] 
       
    @classmethod
    def decode_yolo(cls, bboxes, device):
        bz = bboxes.size(0)
        idxs_i = torch.arange(cls.S)
        idxs_j = torch.arange(cls.S)
        pos_j, pos_i = torch.meshgrid(idxs_i, idxs_j, indexing='ij')
        pos_i = pos_i.expand((bz, -1, -1)).unsqueeze(3).expand((-1, -1, -1, 2))
        pos_j = pos_j.expand((bz, -1, -1)).unsqueeze(3).expand((-1, -1, -1, 2))
        pos_i = pos_i.to(device)
        pos_j = pos_j.to(device)
        xc = (bboxes[..., 0] + pos_i) / cls.S
        yc = (bboxes[..., 1] + pos_j) / cls.S
        x1 = torch.clamp(xc - bboxes[..., 2] **2 / 2, min=0)
        y1 = torch.clamp(yc - bboxes[..., 3] **2 / 2, min=0)
        x2 = torch.clamp(xc + bboxes[..., 2] **2 / 2, max=1)
        y2 = torch.clamp(yc + bboxes[..., 3] **2 / 2, max=1)

        return torch.stack((x1, y1, x2, y2) ,dim=-1)
    
    @classmethod
    def reshape_data(cls,data):
        data_cls = data[..., 10:]
        data_conf = data[..., 8:10].reshape((-1, cls.S, cls.S, cls.B, 1))
        data_bboxes = data[..., :8].reshape((-1, cls.S, cls.S, cls.B, 4))
        return data_bboxes, data_conf, data_cls
    
    @classmethod
    def to_numpy(cls, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.array):
            return data
        else:
            raise Exception(f"{data} is a type of {type(data)}, not numpy/tensor type")
    
    @classmethod
    def image_to_numpy(cls, image):
        if isinstance(image, torch.Tensor):
            image = image.squeeze().detach().cpu().numpy()
            image = image.transpose((1, 2, 0))
            image = Unnormalize()(image)
            image = np.ascontiguousarray(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        elif isinstance(image, np.array):
            return image
        else:
            raise Exception(f"{image} is a type of {type(image)}, not numpy/tensor type")
        
    @classmethod
    def nms(self, pred_bboxes, pred_confs, pred_cls, iou_thresh):
        idxs = torchvision.ops.nms(pred_bboxes, pred_confs, iou_thresh)
        nms_bboxes = pred_bboxes[idxs]
        nms_confs = pred_confs[idxs]
        nms_classes = pred_cls[idxs]

        return nms_bboxes, nms_confs, nms_classes

