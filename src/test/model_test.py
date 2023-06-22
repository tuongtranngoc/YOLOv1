import cv2
import torch
import numpy as np
from torchview import draw_graph
from ..models.modules.yolo import YoloModel


def test_YOLO_model():
    print('### Checking YOLO model')
    input_size = 448
    num_classes = 20
    inp = torch.randn(2, 3, input_size, input_size)
    device = torch.device('cpu')
    model = YoloModel(input_size=input_size, backbone="vgg16", num_classes=num_classes, pretrained=True).to('cpu')
    # summary(model)
    graph = draw_graph(model, input_size=inp.shape, expand_nested=True, save_graph=True, directory='exps', graph_name='model')


test_YOLO_model()
