from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_utils import *
from ..data import CFG


class SumSquaredError(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5) -> None:
        super(SumSquaredError, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.cfg = CFG
        self.mse_loss_fn = nn.MSELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, gt, pred):
        bz = gt.size(0)
        S = self.cfg['S']
        B = self.cfg['B']
        C = self.cfg['C']
        gt_bboxes, gt_conf, gt_cls = BoxUtils.reshape_data(gt)
        pred_bboxes, pred_conf, pred_cls = BoxUtils.reshape_data(pred)

        gt_bboxes = gt_bboxes.clone()
        ious = compute_iou(gt_bboxes, pred_bboxes, self.device)
        _, max_idxs = torch.max(ious, dim=-1)
        
        one_obj_ij = (gt_conf[..., 0] == 1)
        one_obj_i = (gt_conf[..., 0] == 1)[..., 0]

        idxs = torch.where(one_obj_i==True)
        for bz, j, i in zip(*idxs):
            bz, j, i = bz.item(), j.item(), i.item()
            max_id = max_idxs[bz, j, i]
            one_obj_ij[bz, j, i, 1-max_id] = False
            gt_conf[bz, j, i, 1-max_id, 0] = 0

        one_noobj_ij = ~one_obj_ij
        
        box_loss = self.mse_loss_fn(pred_bboxes[one_obj_ij], gt_bboxes[one_obj_ij])

        obj_loss = self.mse_loss_fn(pred_conf[..., 0][one_obj_ij], gt_conf[..., 0][one_obj_ij])

        noobj_loss = self.mse_loss_fn(pred_conf[..., 0][one_noobj_ij], gt_conf[..., 0][one_noobj_ij])
         
        cls_loss = self.mse_loss_fn(pred_cls[one_obj_i], gt_cls[one_obj_i])

        box_loss = self.lambda_coord  * box_loss

        conf_loss = (self.lambda_noobj * noobj_loss + obj_loss)
        
        return box_loss / bz, conf_loss / bz, cls_loss / bz