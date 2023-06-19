import torch
from tqdm import tqdm

from .config import CFG
from .utils.torch_utils import *
from .utils.metrics import BatchMeter
from .utils.losses import SumSquaredError
from .utils.visualization import Vizualization

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.loss_fn = SumSquaredError().to(self.device)
        self.dataloader = DataLoader(
            self.dataset,
            self.bz,
            self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_men)

    def cal_mAP(self, map_mt, pred_bbox, pred_conf, pred_cls, gt_bbox, gt_conf, gt_cls):
        """https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html"""
        preds = [{"boxes": pred_bbox.to(), "scores": pred_conf, "labels": pred_cls}]

        target = [{"boxes": gt_bbox, "scores": gt_conf, "labels": gt_cls}]

        map_mt.update(preds, target)

    def evaluate(self):
        metrics = {
            "eval_box_loss": BatchMeter("box_loss"),
            "eval_conf_loss": BatchMeter("conf_loss"),
            "eval_cls_loss": BatchMeter("cls_loss"),
            "eval_map": BatchMeter("map"),
            "eval_map_50": BatchMeter("map_50"),
            "eval_map_75": BatchMeter("map_75"),
        }

        map_mt = MeanAveragePrecision(class_metrics=True)

        self.model.eval()

        for i, (images, labels) in enumerate(tqdm(self.dataloader)):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                out = self.model(images)
                box_loss, conf_loss, cls_loss = self.loss_fn(labels, out)

                metrics["eval_box_loss"].update(box_loss)
                metrics["eval_conf_loss"].update(conf_loss)
                metrics["eval_cls_loss"].update(cls_loss)

                for j in range(images.size(0)):
                    pred_bboxes, pred_conf, pred_cls = Vizualization.reshape_data(out[j].unsqueeze(0))
                    gt_bboxes, gt_conf, gt_cls = Vizualization.reshape_data(labels[j].unsqueeze(0))

                    pred_bboxes, pred_conf, pred_cls = BoxUtils.nms(
                        pred_bboxes,
                        pred_conf,
                        pred_cls,
                        cfg["iou_thresh"],
                        cfg["conf_thresh"])
                    self.cal_mAP(
                        map_mt,
                        pred_bboxes,
                        pred_conf,
                        pred_cls,
                        gt_bboxes,
                        gt_conf,
                        gt_cls)

        mAP = map_mt.compute()
        metrics["eval_map"].update(mAP["map"])
        metrics["eval_map_50"].update(mAP["map_50"])
        metrics["eval_map_75"].update(mAP["map_75"])

        print(f'[EVALUATE] - box_loss: {metrics["eval_box_loss"].get_value("mean"):.5f}, \
            conf_loss: {metrics["eval_conf_loss"].get_value("mean"):.5f}, \
            cls_loss: {metrics["eval_cls_loss"].get_value("mean"):.5f}')

        print(f'[EVALUATE] - map: {metrics["eval_map"].get_value("mean"):.5f}, \
            map_50: {metrics["eval_map_50"].get_value("mean"):.5f}, \
            map_75: {metrics["eval_map_75"].get_value("mean"):.5f}')

        return metrics
