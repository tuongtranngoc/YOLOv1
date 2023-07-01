# This project aims to implement YOLOV1 algorithm

## Introduction
YOLOv1 is new approach to object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. YOLO is extremely fast, YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance.

## Update news
+ `2023/07/02`: Update GIoU, DIoU, CIoU
+ `2023/07/01`: Update model weights Resnet18, Resnet34, Resnet50


## Experiment Tables

| Backbone | Dataset | Training dataset | Valid dataset | Image size | mAP | mAP_50 | mAP_75 | Files |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG16|PASCAL VOC|trainval 2007+2012|test2007|448x448|--|--|--|--|
|ResNet18|PASCAL VOC|trainval 2007+2012|test2007|448x448|0.31|0.55|0.31|--|
|ResNet34|PASCAL VOC|trainval 2007+2012|test2007|448x448|0.35|0.58|0.35|--|
|ResNet50|PASCAL VOC|trainval 2007+2012|test2007|448x448|0.32|0.60|0.31|--|

## Dataset
+ Download Pascal VOC train+val 2012+2007
+ Download Pascal VOC test 2007

Put all images, annotations, txt files in folder `dataset/VOC` folder as following:

```shell
├── VOC
    ├── images
        ├── trainval2007
            ├── 000005.jpg
            ├── 000007.jpg
        ├── trainval2012
        ├── test2007
    ├── images_id
        ├── trainval2007.txt
        ├── trainval2012.txt
        ├── test2007.txt
    ├── labels
        ├── trainval2007
            ├── 000005.xml
            ├── 000007.xml
        ├── trainval2012
        ├── trainval2007
```

### Training
```shell
python -m src.train --model_type resnet18/resnet34/resnet50 --resume resume_most_recent_training
```
### Evaluate
```shell
python -m src.eval --model_type resnet18/resnet34/resnet50 --weight_type path_to_weight_best.pt
```