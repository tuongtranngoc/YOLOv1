import cv2
import torch
import numpy as np
from data.augmentation import *
from ..data.utils import *
from torch.utils.data import DataLoader
from ..data.dataset_yolo import YoloDatset
from ..utils.torch_utils import BoxUtils

aug = AlbumAug()

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


def debug_dataset(images, labels):
    labels = labels.unsqueeze(3).expand((-1, -1, -1, 2, -1))
    labels = labels.clone()
    labels[..., :4] = BoxUtils.decode_yolo(labels[..., :4], device)
    for i in range(images.size(0)):
        image, target = images[i], labels[i]
        mask = (target[..., 4] > 0)
        target = target[mask]

        image = image.detach().cpu().numpy()
        image = image.transpose((1, 2, 0))
        image = Unnormalize()(image)
        image = np.ascontiguousarray(image, dtype=np.uint8)
    
        target = target.detach().cpu().numpy()
        target[..., :4] *= image_size
        target = target.astype(np.int32)
        for box in target:
            cls = np.argmax(box[10:])
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
            image = cv2.putText(image, str(id2classes[cls+1]),  (box[0], box[1]+15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(255, 0, 0))
        os.makedirs('exps/dataset', exist_ok=True)
        cv2.imwrite(f'exps/dataset/{i}.jpg', image)


debug_dataset(imgs, labels)