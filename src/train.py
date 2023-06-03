from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src import CFG
from src.utils.visualization import *

from .data.utils import *
from .utils.metrics import BatchMeter
from .models.model_yolo import YOLOv1
from .utils.visualization import Debuger
from .data.dataset_yolo import YoloDatset
from .utils.losses import SumSquaredError
from .utils.tensorboard import Tensorboard


class Trainer:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = CFG
        self.create_model()
        self.create_dataloader()
        self.debuger = Debuger(self.cfg['training_debug'])

    def create_dataloader(self):
        self.train_dataset = YoloDatset(self.cfg['VOC']['image_path'], self.cfg['VOC']['anno_path'], self.cfg['VOC']['txt_train_path'])
        self.val_dataset = YoloDatset(self.cfg['VOC']['image_path'], self.cfg['VOC']['anno_path'], self.cfg['VOC']['txt_val_path'])
        print("Creating dataloader ...")
        self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(self.val_dataset, batch_size=4, shuffle=False, num_workers=8)

    def create_model(self):
        print("Creating model ...")
        self.model = YOLOv1().to(self.device)
        print("Creating loss function ...")
        self.loss = SumSquaredError().to(self.device)
        print("Creating optimzer ...")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, amsgrad=True)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10)

    def train(self):
        mt_box_loss = BatchMeter('box_loss', 'mean')
        mt_conf_loss = BatchMeter('conf_loss', 'mean')
        mt_cls_loss = BatchMeter('cls_loss', 'mean')

        for epoch in range(self.cfg['epochs']):
            for bz, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()
                batch_size = images.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device)
                out = self.model(images)

                box_loss, conf_loss, class_loss = self.loss_fn(labels, out)
                total_loss = box_loss + conf_loss + class_loss
                
                total_loss.backward()
                self.optimizer.step()

                mt_box_loss.update(box_loss.item())
                mt_conf_loss.update(conf_loss.item())
                mt_cls_loss.update(class_loss.item())

                me = self.metrics.compute()
                print(f'Epoch {epoch+1} Batch {bz+1}/{len(self.train_loader)}, \
                    box_loss: {mt_box_loss.get_value(): .5f}, \
                        conf_loss: {mt_conf_loss.get_value():.5f}, \
                            class_loss: {mt_cls_loss.get_value():.5f}', end='\r')
            
            # Debug image at each training time
            with torch.no_grad():
                self.debuger.debug_output(self.train_dataset, self.cfg['idxs_debug'], self.model, 'train', self.device, self.cfg['conf_debug'])
                self.debuger.debug_output(self.val_dataset, self.cfg['idxs_debug'], self.model, 'val', self.device, self.cfg['conf_debug'])
            
            Tensorboard.update_metric('train', epoch, box_loss=mt_box_loss.get_value(), conf_loss=mt_conf_loss.get_value(), class_loss=mt_cls_loss.get_value())
            print(f"TRAIN: box_loss: {me.box_loss: .5f}, conf_loss: {conf_loss: .5f}, class_loss: {class_loss: .5f}")
        
    def save_ckpt(self, save_path):
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": 0
        }
        torch.save(ckpt_dict, save_path)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
