import os
import torch
import torchvision
from tqdm import tqdm

from .config import CFG
from .utils.torch_utils import *
from .utils.metrics import BatchMeter
from .utils.visualization import Debuger
from .utils.losses import SumSquaredError
from .utils.tensorboard import Tensorboard

import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class VocEval:
    def __init__(self, dataset, model, bz, shuffle, num_workers, pin_men):
        super().__init__()
        self.bz = bz
        self.cfg = CFG
        self.model = model
        self.dataset = dataset
        self.shuffle = shuffle
        self.pin_men = pin_men
        self.num_workers = num_workers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.loss_fn = SumSquaredError().to(self.device)
        self.map = MeanAveragePrecision(iou_type="bbox")
        self.dataloader = DataLoader(self.dataset, self.bz, self.shuffle, num_workers=self.num_workers, pin_memory=self.pin_men)

    def cal_mAP(self, pred_bbox, pred_conf, pred_cls, gt_bbox, gt_conf, gt_cls):
        """https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
        """
        preds = [
            {
                "boxes": pred_bbox,
                "scores": pred_conf,
                "labels": pred_cls
            }
        ]

        target = [
            {
                "boxes": gt_bbox,
                "scores": gt_conf,
                "labels": gt_cls
            }
        ]

        self.map.update(preds, target)
        result = self.map.compute()
        return result

    def evaluate(self):
        eval_box_loss = BatchMeter('box_loss')
        eval_conf_loss = BatchMeter('conf_loss')
        eval_cls_loss = BatchMeter('cls_loss')
        self.model.eval()

        for i, (images, labels) in enumerate(tqdm(self.dataloader)):
            batch_size = images.size(0)
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                out = self.model(images)
                bpred_bboxes, bpred_conf, bpred_cls = BoxUtils.reshape_data(out)
                bpred_bboxes = BoxUtils.decode_yolo(bpred_bboxes, self.device)
                bpred_cls = torch.argmax(bpred_cls, dim=-1)

                btarget_bboxes, btarget_conf, btarget_cls = BoxUtils.reshape_data(labels)
                btarget_bboxes = BoxUtils.decode_yolo(btarget_bboxes, self.device)
                btarget_cls = torch.argmax(btarget_cls, dim=-1)

                box_loss, conf_loss, cls_loss = self.loss_fn(labels, out)

                eval_box_loss.update(box_loss)
                eval_conf_loss.update(conf_loss)
                eval_cls_loss.update(cls_loss)
            
        return eval_box_loss, eval_conf_loss, eval_cls_loss
    
            # for j in range(batch_size):
            #     pred_bboxes = bpred_bboxes[j].reshape((-1, 4))
            #     pred_conf = bpred_conf[j].reshape(-1)
            #     pred_cls = bpred_cls[j].unsqueeze(-1).expand((-1, -1, 2)).reshape(-1)

            #     target_bboxes = btarget_bboxes[j][..., 0, :].reshape((-1, 4))
            #     target_conf = btarget_conf[j][..., 0, :].squeeze(-1).reshape(-1)
            #     target_cls = btarget_cls[j].reshape(-1)

            #     obj_mask = torch.where(target_conf > 0)
            #     target_bboxes = target_bboxes[obj_mask]
            #     target_conf = target_conf[obj_mask]
            #     target_cls = target_cls[obj_mask]

            #     pred_bboxes, pred_conf, pred_cls = BoxUtils.nms(pred_bboxes, pred_conf, pred_cls, self.cfg['iou_thresh'])

            #     map = self.cal_mAP(pred_bboxes, pred_conf, pred_cls, target_bboxes, target_conf, target_cls)

            

