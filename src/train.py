from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.utils.data import DataLoader

from src import CFG
from src.eval import VocEval
from src.utils.visualization import *

from .data.utils import *
from .utils.metrics import BatchMeter
from .utils.visualization import Debuger
from .data.dataset_yolo import YoloDatset
from .utils.losses import SumSquaredError
from .models.modules.yolo import YoloModel
from .utils.tensorboard import Tensorboard


class Trainer:
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = CFG
        self.best_map = 0
        self.create_model()
        self.create_dataloader()
        self.debuger = Debuger(self.cfg["training_debug"])
        self.eval = VocEval(
            self.val_dataset,
            self.model,
            self.cfg["bz_valid"],
            False,
            self.cfg["n_workers"],
            False)

    def create_dataloader(self):
        self.train_dataset = YoloDatset(
            self.cfg["VOC"]["image_path"],
            self.cfg["VOC"]["anno_path"],
            self.cfg["VOC"]["txt_train_path"],
            is_augment=True)
        self.val_dataset = YoloDatset(
            self.cfg["VOC"]["image_path"],
            self.cfg["VOC"]["anno_path"],
            self.cfg["VOC"]["txt_val_path"])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg["bz_train"],
            shuffle=True,
            num_workers=self.cfg["n_workers"],
            pin_memory=False)

    def create_model(self):
        self.model = YoloModel(
            input_size=self.cfg["image_size"][0],
            backbone="vgg16",
            num_classes=self.cfg["C"],
            pretrained=True,).to(self.device)
        self.loss_fn = SumSquaredError().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, weight_decay=5e-4, lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [75, 105], gamma=0.1)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, amsgrad=True)

    def train(self):
        for epoch in range(1, self.cfg["epochs"]):
            mt_box_loss = BatchMeter()
            mt_conf_loss = BatchMeter()
            mt_cls_loss = BatchMeter()

            for bz, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)
                out = self.model(images)

                box_loss, conf_loss, class_loss = self.loss_fn(labels, out)
                total_loss = box_loss + conf_loss + class_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                mt_box_loss.update(box_loss.item())
                mt_conf_loss.update(conf_loss.item())
                mt_cls_loss.update(class_loss.item())

                print(f"Epoch {epoch} Batch {bz+1}/{len(self.train_loader)}, box_loss: {mt_box_loss.get_value(): .5f}, conf_loss: {mt_conf_loss.get_value():.5f}, class_loss: {mt_cls_loss.get_value():.5f}",
                        end="\r")

                Tensorboard.add_scalars(
                    "train",
                    epoch,
                    box_loss=mt_box_loss.get_value(),
                    conf_loss=mt_conf_loss.get_value(),
                    cls_loss=mt_cls_loss.get_value())

            print(f"[TRAIN] - Epoch: {epoch} - box_loss: {mt_box_loss.get_value('mean'): .5f}, conf_loss: {mt_conf_loss.get_value('mean'): .5f}, class_loss: {mt_cls_loss.get_value('mean'): .5f}")

            if epoch % self.cfg["eval_step"] == 0:
                metrics = self.eval.evaluate()
                Tensorboard.add_scalars(
                    "eval_loss",
                    epoch,
                    box_loss=metrics["eval_box_loss"].get_value("mean"),
                    conf_loss=metrics["eval_conf_loss"].get_value("mean"),
                    cls_loss=metrics["eval_cls_loss"].get_value("mean"))

                Tensorboard.add_scalars(
                    "eval_map",
                    epoch,
                    mAP=metrics["eval_map"].get_value("mean"),
                    mAP_50=metrics["eval_map_50"].get_value("mean"),
                    mAP_75=metrics["eval_map_75"].get_value("mean"))

                if metrics["eval_map"].get_value("mean") > self.best_map:
                    self.best_map = metrics["eval_map"].get_value("mean")
                    self.save_ckpt(self.cfg["best_ckpt_path"], self.best_map, epoch)
                self.save_ckpt(self.cfg["last_ckpt_path"], self.best_map, epoch)

            # Debug image at each training time
            with torch.no_grad():
                self.debuger.debug_output(
                    self.train_dataset,
                    self.cfg["idxs_debug"],
                    self.model,
                    "train",
                    self.device,
                    self.cfg["conf_debug"],
                    False,
                )
                self.debuger.debug_output(
                    self.val_dataset,
                    self.cfg["idxs_debug"],
                    self.model,
                    "val",
                    self.device,
                    self.cfg["conf_debug"],
                    True,
                )

    def save_ckpt(self, save_path, best_acc, epoch):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": best_acc,
            "epoch": epoch,
        }
        torch.save(ckpt_dict, save_path)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
