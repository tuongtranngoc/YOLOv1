import cv2
import torch
import numpy as np
from ..data.utils import *
from ..config import CFG as cfg
from torch.utils.data import DataLoader
from ..utils.torch_utils import BoxUtils
from ..data.augmentation import AlbumAug
from ..data.dataset_yolo import YoloDatset

ds = YoloDatset('dataset/VOC2012/JPEGImages', 'dataset/VOC2012/Annotations', 'dataset/VOC2012/ImageSets/Main/val.txt')
val_dataloader = DataLoader(ds, batch_size=10, shuffle=True)

imgs, labels = next(iter(val_dataloader))

id_map = json.load(open('dataset/VOC2012/label_to_id.json'))
id2classes = {
    id_map[k]: k
    for k in id_map.keys()
}

grid_size = 7
device = 'cpu'
image_size = 448

aug = AlbumAug()

def debug_augmentation(images, labels):
    labels = labels.unsqueeze(3).expand((-1, -1, -1, 2, -1))
    labels = labels.permute(0, 2, 1, 3, 4)
    labels = labels.clone()
    labels[..., :4] = BoxUtils.decode_yolo(labels[..., :4], device)
    
    for i in range(images.size(0)):
        image, target = images[i], labels[i]
        mask = (target[..., 4] > 0)
        target = target[mask]

        bboxes = target[:, :4]
        bboxes = BoxUtils.to_numpy(bboxes)
        bboxes = (bboxes*image_size).astype(np.float32)

        cls_labels = target[:, 10:]
        cls_labels = BoxUtils.to_numpy(cls_labels)
        cls_labels = np.argmax(cls_labels, axis=-1)
        image = BoxUtils.image_to_numpy(image)
        image, bboxes, cls_labels = aug(image, bboxes, cls_labels)
        
        bboxes = bboxes.astype(np.int32)
        for box, label in zip(bboxes, cls_labels):
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
            image = cv2.putText(image, str(id2classes[label+1]), \
                                (box[0], box[1]+15), \
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
                                fontScale=0.5, thickness=1, color=(255, 0, 0))
        os.makedirs(cfg['augmentation_debug'], exist_ok=True)
        cv2.imwrite(f"{cfg['augmentation_debug']}/{i}.jpg", image)


debug_augmentation(imgs, labels)