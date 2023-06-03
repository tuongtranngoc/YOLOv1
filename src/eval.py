import os
import torch

from .config import CFG
from .utils.torch_utils import *
from .utils.metrics import BatchMetric, BatchMeter
from .utils.visualization import Debuger
from .utils.losses import SumSquaredError

import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision



class VocEval:
    """https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
    """
    def __init__(self, dataset, model, bz, shuffle, num_workers, pin_men):
        self.cfg = CFG
        self.bz = bz
        self.model = model
        self.dataset = dataset
        self.shuffle = shuffle
        self.pin_men = pin_men
        self.num_workers = num_workers

        self.loss_fn = SumSquaredError()
        self.metric = MeanAveragePrecision(iou_type="bbox")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataloader = DataLoader(self.dataset, self.bz, self.shuffle, num_workers=self.num_workers, pin_memory=self.pin_men)

    def cal_mAP(self, pred_bbox, pred_conf, pred_cls, gt_bbox, gt_conf, gt_cls):
        preds = [
            {
                "bboxes": pred_bbox,
                "scores": pred_conf,
                "labels": pred_cls
            }
        ]

        target = [
            {
                "bboxes": gt_bbox,
                "scores": gt_conf,
                "labels": gt_cls
            }
        ]

        self.metric.update(preds, target)
        result = self.metric.compute()
        return result
    
    def cal_loss(self, box_loss, conf_loss, cls_loss):
        pass


    def evaluate(self):
        mt_mAP = BatchMeter('map')
        for i, (images, labels) in enumerate(self.dataloader):
            self.model.eval()
            batch_size = images.size(0)
            images = images.to(self.device)
            labels = labels.to(self.device)
            out = self.model(images)

            bpred_bboxes, bpred_conf, bpred_cls = Decode.reshape_data(out)
            bpred_bboxes = Decode.decode_yolo(bpred_bboxes)
            bpred_cls = torch.argmax(bpred_cls, dim=-1)

            btarget_bboxes, btarget_conf, btarget_cls = Decode.reshape_data(labels)
            btarget_bboxes = Decode.decode_yolo(btarget_bboxes)
            btarget_cls = torch.argmax(bpred_cls, dim=-1)

            for j in batch_size:
                pred_bboxes = bpred_bboxes[j].reshape((-1, 4))
                pred_conf = bpred_conf[j].reshape((-1, 1))
                pred_cls = bpred_cls[j]

                target_bboxes = btarget_bboxes[j].reshape((-1, 4))
                target_conf = btarget_conf[j].reshape((-1, 1))
                target_cls = btarget_cls[j]





            

