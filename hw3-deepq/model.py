# 2018 HTC Corporation. All Rights Reserved.
#
# This source code is licensed under the HTC license which can be found in the
# LICENSE file in the root directory of this work.

from torchvision import models as models
import torch.nn as nn
import torch

def resnet50(pretrained):
    model = models.resnet50(pretrained=False)

    if pretrained:
        model.load_state_dict(torch.load('pre-trained/resnet50_model.bin'))

    model.fc = nn.Linear(model.fc.in_features, 5)
    return model

def get_model(name, pretrained):
    assert name in ['resnet50', 'mobilenetv2']
    if name == 'resnet50':
        return resnet50(pretrained)
    else:
        import mobilenetv2 as mnet2
        return mnet2.MobileNetV2(pretrained)
