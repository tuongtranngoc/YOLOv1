import torch
import torch.nn as nn

from ..data.utils import *
from ..utils.torch_utils import *
from ..models.model_yolo import YOLOv1

from torchsummary import summary

print("Creating model ...")
model = YOLOv1()

image = torch.randn(2, 3, 448, 448)  # torch.Size([2, 3, 448, 448])
print(summary(model, image))

