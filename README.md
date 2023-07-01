# This project aims to implement YOLOV1 algorithm

## Introduction
YOLOv1 is new approach to object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. YOLO is extremely fast, YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance.

### 1. Experiment Tables

| Backbone | Dataset | Training dataset | Valid dataset | Image size | mAP | mAP_50 | mAP_75 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG16|PASCAL VOC|trainval 2007+2012|test2007|448x448|--|--|--|
|ResNet18|PASCAL VOC|trainval 2007+2012|test2007|448x448|0.31|0.55|0.31|
|ResNet34|PASCAL VOC|trainval 2007+2012|test2007|448x448|0.35|0.58|0.35|