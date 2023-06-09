import os
import json

import cv2
import torch
from ..data.utils import *
from .torch_utils import *
from ..data import CFG as cfg


def class2color(class_name):
    VOC_CLASS2COLOR = {
        'aeroplane': (128, 0, 0),
        'bicycle': (0, 128, 0),
        'bird': (128, 128, 0),
        'boat': (0, 0, 128),
        'bottle': (128, 0, 128),
        'bus': (0, 128, 128),
        'car': (128, 0, 128),
        'cat': (355, 255, 0),
        'chair': (192, 0, 0),
        'cow': (64, 128, 0),
        'diningtable': (192, 128, 0),
        'dog': (64, 0, 128),
        'horse': (192, 0, 128),
        'motorbike': (64, 128, 128),
        'person': (222, 222, 222),
        'pottedplant': (0, 64, 0),
        'sheep': (128, 64, 0),
        'sofa': (0, 192, 0),
        'train': (128, 192, 0),
        'tvmonitor': (0, 64, 128),
        'background' : (128, 128, 128),
        'groundtruth': (0, 0, 255)
    }
    return VOC_CLASS2COLOR[class_name]


class Drawer:
    def __init__(self, image, is_impt=True, type_label='gt') -> None:
        id_map = json.load(open(cfg['VOC']['label2id']))
        self.id2classes = {
            id_map[k]: k
            for k in id_map.keys()
        }
        self.is_impt = is_impt
        self.type_label = type_label
        self.image = image
        self.lw = 1

    def unnormalize_bboxes(self, bbox:list):
        return [b * cfg['image_size'][0] for b in bbox]

    def draw_box_label(self, bbox, conf, label):
        _bbox = self.unnormalize_bboxes(bbox)
        _label = self.id2classes[label+1]
    
        if self.type_label == 'gt': 
            color = class2color('groundtruth')
            _text = _label
        else:
            _text = _label + '-' + str(round(conf, 3))
            color = class2color(_label) if self.is_impt else class2color('background')

        cv2.rectangle(self.image, \
                    (int(_bbox[0]), int(_bbox[1])), \
                    (int(_bbox[2]), int(_bbox[3])), \
                    color=color, \
                    thickness=1, \
                    lineType=cv2.LINE_AA)

        cv2.putText(self.image,
                    _text,
                    (int(_bbox[0]), int(_bbox[1]+0.025*cfg['image_size'][0])),
                    0,
                    self.lw / 3,
                    color=color,
                    thickness=self.lw,
                    lineType=cv2.LINE_AA)
        
        return self.image


class Debuger:
    def __init__(self, save_debug_path) -> None:
        self.S = cfg['S']
        self.B = cfg['B']
        self.C = cfg['C']
        self.cfg = cfg
        self.save_debug_path = save_debug_path

    def debug_output(self, dataset, idxs, model, type_infer, device, conf_thresh, apply_mns=False):
        os.makedirs(f'{self.save_debug_path}/{type_infer}', exist_ok=True)
        model.eval()
        images, targets = [], []
        for index in idxs:
            image, target = dataset[index]
            images.append(image)
            targets.append(target)

        targets = torch.stack(targets, dim=0).to(device)
        images = torch.stack(images, dim=0).to(device)
        
        pred = model(images)

        for i in range(images.size(0)):
            gt_bboxes, gt_conf, gt_cls = Vizualization.reshape_data(targets[i].unsqueeze(0))
            pred_bboxes, pred_conf, pred_cls = Vizualization.reshape_data(pred[i].unsqueeze(0))

            if apply_mns is True:
                pred_bboxes, pred_conf, pred_cls = BoxUtils.nms(pred_bboxes, pred_conf, pred_cls, iou_thresh=cfg['iou_thresh'], conf_thresh=cfg['conf_thresh'])

            gt_bboxes, gt_conf, gt_cls = Vizualization.label2numpy(gt_bboxes, gt_conf, gt_cls)
            pred_bboxes, pred_conf, pred_cls = Vizualization.label2numpy(pred_bboxes, pred_conf, pred_cls)
            
            image = images[i]

            image = Vizualization.draw_debug(image, gt_bboxes, gt_conf, gt_cls, conf_thresh, 'gt')
            image = Vizualization.draw_debug(image, pred_bboxes, pred_conf, pred_cls, conf_thresh, 'pred')
            cv2.imwrite(f'{self.save_debug_path}/{type_infer}/{i}.png', image)


class Vizualization:
    S = cfg['S']
    B = cfg['B']
    C = cfg['C']
    save_debug_path = cfg['prediction_debug']
    os.makedirs(save_debug_path, exist_ok=True)

    @classmethod
    def reshape_data(cls, out):
        pred_bboxes = out[..., :8]
        pred_confs = out[..., 8:10].reshape(-1)
        pred_cls = torch.argmax(out[..., 10:], dim=-1).unsqueeze(-1).expand((-1, -1, -1, 2)).reshape(-1)
        pred_bboxes = BoxUtils.decode_yolo(pred_bboxes.reshape(-1, cls.S, cls.S, cls.B, 4)).reshape((-1, 4))

        return pred_bboxes, pred_confs, pred_cls
    
    @classmethod
    def label2numpy(cls, *args):
        args_list = []
        for i in range(len(args)):
            args_list.append(BoxUtils.to_numpy(args[i]))
        return args_list

    @classmethod
    def image2numpy(cls, images):
        images = BoxUtils.image_to_numpy(images)
        return images
    
    @classmethod
    def draw_debug(cls, image, bboxes, confs, classes, conf_thresh, type_draw='pred'):
        image = cls.image2numpy(image)
        bboxes, confs, classes = cls.label2numpy(bboxes, confs, classes)
        for bbox, conf, label in zip(bboxes, confs, classes):
            if conf >= conf_thresh:
                image = Drawer(image, True, type_draw).draw_box_label(bbox, conf, label)
        return image