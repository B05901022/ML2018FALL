# 2018 HTC Corporation. All Rights Reserved.
#
# This source code is licensed under the HTC license which can be found in the
# LICENSE file in the root directory of this work.

import torch

class BottleNeck(torch.nn.Module):
    def __init__(self, in_n, out_n, s, t):
        super(BottleNeck, self).__init__()
        self.residual = (s == 1 and in_n == out_n)
        self.conv1x1_1 = torch.nn.Conv2d(in_n, in_n * t, 1, bias=False)
        self.bn_1 = torch.nn.BatchNorm2d(in_n * t)
        self.relu_1 = torch.nn.ReLU6(inplace=True)
        self.conv3x3 = torch.nn.Conv2d(in_n * t, in_n * t, 3, s, 1, bias=False, groups=in_n * t)
        self.bn_2 = torch.nn.BatchNorm2d(in_n * t)
        self.relu_2 = torch.nn.ReLU6(inplace=True)
        self.conv1x1_2 = torch.nn.Conv2d(in_n * t, out_n, 1, bias=False)
        self.bn_3 = torch.nn.BatchNorm2d(out_n)

    def forward(self, x):
        y = self.conv1x1_1(x)
        y = self.bn_1(y)
        y = self.relu_1(y)
        y = self.conv3x3(y)
        y = self.bn_2(y)
        y = self.relu_2(y)
        y = self.conv1x1_2(y)
        y = self.bn_3(y)
        if self.residual:
            y = x + y
        return y

class avgpool(torch.nn.Module):
    def __init__(self, w, h):
        super(avgpool, self).__init__()
        self.avgpool = torch.nn.AvgPool2d((w, h))
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class MobileNetV2(torch.nn.Module):
    def __init__(self, pretrained):
        super(MobileNetV2, self).__init__()

        num_classes = 1000 if pretrained else 5

        plan = [[1,16,1,1],
                [6,24,2,2],
                [6,32,3,2],
                [6,64,4,2],
                [6,96,3,1],
                [6,160,3,2],
                [6,320,1,1]]

        layers = []
        layers.append(torch.nn.Conv2d(3, 32, 3, 2, 1, bias=False))
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU6(inplace=True))

        ch = 32

        for t, c, n, s in plan:
            for i in range(n):
                if i == 0:
                    layers.append(BottleNeck(ch, c, s, t))
                else:
                    layers.append(BottleNeck(ch, c, 1, t))
                ch = c

        layers.append(torch.nn.Conv2d(ch, 1280, 1, bias=False))
        layers.append(torch.nn.BatchNorm2d(1280))
        layers.append(torch.nn.ReLU6(inplace=True))
        layers.append(avgpool(7, 7))
        self.dropout = torch.nn.Dropout()
        layers.append(self.dropout)
        layers.append(torch.nn.Linear(1280, num_classes))

        self.model = torch.nn.Sequential(*layers)

        if pretrained:
            self.load_state_dict(torch.load('pre-trained/mobilenetv2_model.bin'))
            del self.model[-1]
            self.model.add_module(str(len(self.model)), torch.nn.Linear(1280, 5))

    def set_dropout(self, p):
        self.dropout.p = p

    def forward(self, x):
        return self.model(x)
