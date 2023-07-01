import os
import torch
import argparse
from tqdm import tqdm

from .config import CFG
from .utils.torch_utils import *
from .utils.metrics import BatchMeter
from .data.dataset_yolo import YoloDatset
from .utils.losses import SumSquaredError
from .models.modules.yolo import YoloModel
from .utils.visualization import Vizualization

from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .utils.logger import Logger
logger = Logger.get_logger("EVALUATE")

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
        preds = [{"boxes": pred_bbox, "scores": pred_conf, "labels": pred_cls}]

        target = [{"boxes": gt_bbox, "scores": gt_conf, "labels": gt_cls}]

        map_mt.update(preds, target)

    def evaluate(self):
        metrics = {
            "eval_box_loss": BatchMeter(),
            "eval_conf_loss": BatchMeter(),
            "eval_cls_loss": BatchMeter(),

            "eval_map": BatchMeter(),
            "eval_map_50": BatchMeter(),
            "eval_map_75": BatchMeter(),
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

                    mask_obj = gt_conf > 0
                    gt_bboxes = gt_bboxes[mask_obj]
                    gt_conf = gt_conf[mask_obj]
                    gt_cls = gt_cls[mask_obj]
                    
                    pred_bboxes = pred_bboxes
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
        
        logger.info(f'box_loss: {metrics["eval_box_loss"].get_value("mean"):.5f}, \
            conf_loss: {metrics["eval_conf_loss"].get_value("mean"):.5f}, \
            cls_loss: {metrics["eval_cls_loss"].get_value("mean"):.5f}')

        logger.info(f'mAP: {metrics["eval_map"].get_value("mean"):.5f}, \
            mAP_50: {metrics["eval_map_50"].get_value("mean"):.5f}, \
            mAP_75: {metrics["eval_map_75"].get_value("mean"):.5f}')

        return metrics
    

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_type', type=str, default='best.pt',
                        help='weight type: best.pt/last.pt')
    parser.add_argument('--model_type', type=str, default='resnet34',
                        help='Model selection contain: vgg16, vgg16-bn, resnet18, resnet34')
    parser.add_argument('--bz_eval', type=int, default=cfg['bz_valid'],
                        help='Batch size valid dataset')
    parser.add_argument('--n_workers', type=int, default=cfg['n_workers'],
                        help='Number of workers')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = YoloDatset(
            cfg["VOC"]["image_path"],
            cfg["VOC"]["anno_path"],
            cfg["VOC"]["txt_val_path"])
    model = YoloModel(
            input_size=cfg["image_size"][0],
            backbone=args.model_type,
            num_classes=cfg["C"],
            pretrained=False).to(device)
    ckpt_path = os.path.join(cfg['ckpt_dirpath'], args.model_type, args.weight_type)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    eval = VocEval(dataset, model, args.bz_eval, False, args.n_workers, False)
    eval.evaluate()