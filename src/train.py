from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src import CFG
from src.eval import VocEval
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
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = CFG
        self.best_loss = 0
        self.create_model()
        self.create_dataloader()
        self.debuger = Debuger(self.cfg['training_debug'])
        self.eval = VocEval(self.val_dataset, self.model, self.cfg['bz_valid'], False, self.cfg['n_workers'], False)

    def create_dataloader(self):
        self.train_dataset = YoloDatset(self.cfg['VOC']['image_path'], self.cfg['VOC']['anno_path'], self.cfg['VOC']['txt_train_path'], is_augment=False)
        self.val_dataset = YoloDatset(self.cfg['VOC']['image_path'], self.cfg['VOC']['anno_path'], self.cfg['VOC']['txt_val_path'])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg['bz_train'], shuffle=True, num_workers=self.cfg['n_workers'], pin_memory=False)
        #self.val_loader = DataLoader(self.val_dataset, batch_size=self.cfg['bz_valid'], shuffle=False, num_workers=self.cfg['n_workers'])

    def create_model(self):
        self.model = YOLOv1().to(self.device)
        self.loss_fn = SumSquaredError().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, weight_decay=5e-4, lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [75, 105], gamma=0.1)

    def train(self):
        for epoch in range(1, self.cfg['epochs']):
            mt_box_loss = BatchMeter('box_loss')
            mt_conf_loss = BatchMeter('conf_loss')
            mt_cls_loss = BatchMeter('cls_loss')
            
            for bz, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()
        
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
                
                print(f'Epoch {epoch} Batch {bz+1}/{len(self.train_loader)}, \
                            box_loss: {mt_box_loss.get_value(): .5f}, \
                            conf_loss: {mt_conf_loss.get_value():.5f}, \
                            class_loss: {mt_cls_loss.get_value():.5f}', end='\r')
                
                Tensorboard.add_scalars('train', epoch, box_loss=mt_box_loss.get_value(), \
                                                         conf_loss=mt_conf_loss.get_value(), \
                                                         cls_loss=mt_cls_loss.get_value())

            print(f"TRAIN - Epoch: {epoch} - box_loss: {mt_box_loss.get_value('mean'): .5f}, \
                                            conf_loss: {mt_conf_loss.get_value('mean'): .5f}, \
                                            class_loss: {mt_cls_loss.get_value('mean'): .5f}")
            
            before_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_scheduler.step()
            after_lr = self.optimizer.param_groups[0]["lr"]
            print("Epoch %d: SGD lr %.5f -> %.5f" % (epoch, before_lr, after_lr))

            if epoch % self.cfg['eval_step'] == 0:
                eval_box_loss, eval_conf_loss, eval_cls_loss = self.eval.evaluate()
                print(f'EVALUATE - box_loss: {eval_box_loss.get_value("mean"):.5f}, \
                            conf_loss: {eval_conf_loss.get_value("mean"):.5f}, \
                            cls_loss: {eval_cls_loss.get_value("mean"):.5f}')
                
                Tensorboard.add_scalars('eval', epoch, box_loss=eval_box_loss.get_value('mean'), \
                                                        conf_loss=eval_conf_loss.get_value('mean'), \
                                                        cls_loss=eval_cls_loss.get_value('mean'))
                
                self.save_ckpt(self.cfg['last_ckpt_path'], 0, epoch)

            # Debug image at each training time
            with torch.no_grad():
                self.debuger.debug_output(self.train_dataset, self.cfg['idxs_debug'], self.model, 'train', self.device, self.cfg['conf_debug'])
                self.debuger.debug_output(self.val_dataset, self.cfg['idxs_debug'], self.model, 'val', self.device, self.cfg['conf_debug'])
            
        
    def save_ckpt(self, save_path, best_acc, epoch):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": best_acc,
            "epoch": epoch
        }
        torch.save(ckpt_dict, save_path)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
