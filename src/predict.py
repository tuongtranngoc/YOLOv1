from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

from .config import CFG as cfg
from .utils.visualization import *
from .models.modules.yolo import YoloModel


class Predictor:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(cfg['image_size'][0], cfg['image_size'][1]),
            A.Normalize(),
            ToTensorV2()
        ])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YoloModel(input_size=cfg['image_size'][0], backbone="vgg16", num_classes=cfg['C'], pretrained=True).to(self.device)
        self.model = self.load_weight(self.model, cfg["last_ckpt_path"])

    def predict(self, image_pth):
        image = cv2.imread(image_pth)
        image = self.tranform(image)
        image = image.to(self.device)
        image = image.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            out = self.model(image)[0]
        
        pred_bboxes, pred_conf, pred_cls = Vizualization.reshape_data(out)
        pred_bboxes, pred_conf, pred_cls = BoxUtils.nms(pred_bboxes, pred_conf, pred_cls, cfg['iou_thresh'], cfg['conf_thresh'])
        image = Vizualization.draw_debug(image, pred_bboxes, pred_conf, pred_cls)
        cv2.imwrite(f'{cfg["prediction_debug"]}/{os.path.basename(image_pth)}', image)

    def tranform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        return image['image']

    def load_weight(self, model, weight_path):
        if os.path.exists(weight_path):
            ckpt = torch.load(weight_path)
            model.load_state_dict(ckpt['model'])
            return model
        else:
            raise Exception(f"Path to model {weight_path} not exist")

if __name__ == "__main__":
    predictor = Predictor()
    image_path = "/home/tuongtran/Researching/Object detection/YOLOv1/dataset/VOC/VOCdevkit/VOC2007/JPEGImages/005264.jpg"
    result = predictor.predict(image_path)