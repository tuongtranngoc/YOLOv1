import cv2
import torch
from ..data.utils import *
from .torch_utils import *
from ..data import CFG as cfg

from .tensorboard import Tensorboard


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
        id_map = json.load(open('dataset/VOC2012/label_to_id.json'))
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
        else:
            color = class2color(_label) if self.is_impt else class2color('background')

        cv2.rectangle(self.image, \
                    (int(_bbox[0]), int(_bbox[1])), \
                    (int(_bbox[2]), int(_bbox[3])), \
                    color=color, \
                    thickness=1, \
                    lineType=cv2.LINE_AA)

        cv2.putText(self.image,
                    _label + '-' + str(round(conf, 3)), \
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
        self.save_debug_path = save_debug_path

    def debug_output(self, dataset, idxs, model, type_infer, device, conf_thresh, epoch, with_tsboard=False):
        model.eval()
        images, targets = [], []
        for index in idxs:
            image, target = dataset[index]
            images.append(image)
            targets.append(target)

        targets = torch.stack(targets, dim=0).to(device)
        images = torch.stack(images, dim=0).to(device)

        bgt_bboxes, bgt_conf, bgt_cls = BoxUtils.reshape_data(targets)
        bgt_bboxes = BoxUtils.decode_yolo(bgt_bboxes, device)
        
        pred = model(images)
        bpred_bboxes, bpred_conf, bpred_cls = BoxUtils.reshape_data(pred)
        bpred_bboxes = BoxUtils.decode_yolo(bpred_bboxes, device)

        for i in range(images.size(0)):
            gt_bboxes = bgt_bboxes[i]
            gt_conf = bgt_conf[i]
            gt_cls = bgt_cls[i]

            pred_bboxes = bpred_bboxes[i]
            pred_conf = bpred_conf[i]
            pred_cls = bpred_cls[i]
            pred_cls = pred_cls.unsqueeze(2).expand((-1, -1, self.B, -1))

            mask = (gt_conf[..., 0] == 1)
            gt_conf = gt_conf[mask]
            gt_bboxes = gt_bboxes[mask]
            gt_cls = gt_cls.unsqueeze(2).expand((-1, -1, self.B, -1))
            gt_cls = gt_cls[mask]
            
            gt_bboxes = BoxUtils.to_numpy(gt_bboxes).tolist()
            gt_conf = BoxUtils.to_numpy(gt_conf).astype(np.float32).tolist()
            gt_cls = np.argmax(BoxUtils.to_numpy(gt_cls), axis=-1).tolist()
            
            pred_conf_obj = pred_conf[mask]
            pred_conf_noobj = pred_conf[~mask]
            pred_bb_obj = pred_bboxes[mask]
            pred_bb_noobj = pred_bboxes[~mask]
            pred_cls_obj = pred_cls[mask]
            pred_cls_noobj = pred_cls[~mask]

            pred_bb_obj = BoxUtils.to_numpy(pred_bb_obj).tolist()
            pred_conf_obj = BoxUtils.to_numpy(pred_conf_obj).astype(np.float32).tolist()
            pred_cls_obj = np.argmax(BoxUtils.to_numpy(pred_cls_obj), axis=-1).tolist()

            pred_bb_noobj = BoxUtils.to_numpy(pred_bb_noobj).tolist()
            pred_conf_noobj = BoxUtils.to_numpy(pred_conf_noobj).astype(np.float32).tolist()
            pred_cls_noobj = np.argmax(BoxUtils.to_numpy(pred_cls_noobj), axis=-1).tolist()

            image = BoxUtils.image_to_numpy(images[i])

            if not with_tsboard:
                for pred_box, pred_cls, pred_conf in zip(pred_bb_noobj, pred_cls_noobj, pred_conf_noobj):
                    if pred_conf[0] > conf_thresh:
                        image = Drawer(image, False, 'pred').draw_box_label(pred_box, pred_conf[0], pred_cls)

                for gt_box, gt_cls, gt_conf in zip(gt_bboxes, gt_cls, gt_conf):
                    image = Drawer(image, True, 'gt').draw_box_label(gt_box, gt_conf[0], gt_cls)

                for pred_box, pred_cls, pred_conf in zip(pred_bb_obj, pred_cls_obj, pred_conf_obj):
                    image = Drawer(image, True, 'pred').draw_box_label(pred_box, pred_conf[0], pred_cls)

                cv2.imwrite(f'{self.save_debug_path}/{type_infer}/{i}.png', image)
            
            else:
                bboxes = pred_bb_noobj + pred_bb_obj
                bboxes = np.array([b * 448 for b in bboxes])
                classes = pred_cls_noobj + pred_cls_obj
                id_map = json.load(open('dataset/VOC2012/label_to_id.json'))
                id2classes = {
                    id_map[k]: k
                    for k in id_map.keys()
                }
                labels = [id2classes[i+1] for i in classes]
                Tensorboard.add_debug_images(os.path.join(type_infer, str(i)), image, bboxes, labels, epoch)