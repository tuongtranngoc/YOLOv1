import torch
import torch.nn as nn

from ..data import CFG as cfg
from .model_base import GlobalAvgPool2d, Conv_BatchNorm_LRelu

S = cfg['S']
B = cfg['B']
C = cfg['C']

class BACKBONE(nn.Module):
    def __init__(self, num_classes=1000):
        super(BACKBONE, self).__init__()

        self.features = nn.Sequential(
            Conv_BatchNorm_LRelu(3, 64, 7, 2),
            nn.MaxPool2d(2, 2),

            Conv_BatchNorm_LRelu(64, 192, 3),
            nn.MaxPool2d(2, 2),

            Conv_BatchNorm_LRelu(192, 128, 1),
            Conv_BatchNorm_LRelu(128, 256, 3),
            Conv_BatchNorm_LRelu(256, 256, 1),
            Conv_BatchNorm_LRelu(256, 512, 3),
            nn.MaxPool2d(2, 2),

            Conv_BatchNorm_LRelu(512, 256, 1),
            Conv_BatchNorm_LRelu(256, 512, 3),

            Conv_BatchNorm_LRelu(512, 256, 1),
            Conv_BatchNorm_LRelu(256, 512, 3),

            Conv_BatchNorm_LRelu(512, 256, 1),
            Conv_BatchNorm_LRelu(256, 512, 3),

            Conv_BatchNorm_LRelu(512, 256, 1),
            Conv_BatchNorm_LRelu(256, 512, 3),

            Conv_BatchNorm_LRelu(512, 512, 1),
            Conv_BatchNorm_LRelu(512, 1024, 3),
            nn.MaxPool2d(2, 2),

            Conv_BatchNorm_LRelu(1024, 512, 1),
            Conv_BatchNorm_LRelu(512, 1024, 3),
            Conv_BatchNorm_LRelu(1024, 512, 1),
            Conv_BatchNorm_LRelu(512, 1024, 3)
        )

        self.classifier = nn.Sequential(
            *self.features,
            GlobalAvgPool2d(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class HEAD(nn.Module):
    def __init__(self):
        super(HEAD, self).__init__()

        self.Conv_BatchNorm_LRelu = nn.Sequential(
            Conv_BatchNorm_LRelu(1024, 1024, 3),
            Conv_BatchNorm_LRelu(1024, 1024, 3, 2),
            Conv_BatchNorm_LRelu(1024, 1024, 3),
            Conv_BatchNorm_LRelu(1024, 1024, 3)
        )

        self.detect = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Conv_BatchNorm_LRelu(x)
        x = self.detect(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()

        self.features = BACKBONE().features
        self.head = HEAD()

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)

        x = x.view(-1, S, S, 5 * B + C)
        return x