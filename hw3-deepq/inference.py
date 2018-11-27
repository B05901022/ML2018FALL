# 2018 HTC Corporation. All Rights Reserved.
#
# This source code is licensed under the HTC license which can be found in the
# LICENSE file in the root directory of this work.

import torch
import model as m
import dataloader as loader
import datetime
from torchvision import transforms
import train
import copy
import numpy as np
import sys
import csv

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ('usage: %s model.bin' % sys.argv[0])
        exit(0)

    model_spec = train.get_model_spec()

    assert "model_name" in model_spec and "pretrained" in model_spec
    assert isinstance(model_spec["pretrained"], bool)
    assert isinstance(model_spec["model_name"], str)

    model = m.get_model(model_spec["model_name"], False).cuda()

    model.load_state_dict(torch.load(sys.argv[1]))

    eval_spec = train.get_eval_spec()
    assert "transform" in eval_spec and "batchsize" in eval_spec
    assert int(eval_spec["batchsize"]) > 0 and int(eval_spec["batchsize"]) <= 64

    print ('[%s] inference start' % (datetime.datetime.now().ctime()))

    model.eval()

    filenames = []
    with open('data/test.csv') as f:
        for r in csv.reader(f):
            filenames.append(r[0])

    writer = csv.writer(open(model_spec['model_name'] + '_prediction_result.csv', 'w'))
    idx = 0
    with torch.no_grad():
        for batch in loader.get_dataloader('test', eval_spec["transform"], int(eval_spec["batchsize"])):
            image = batch['image'].cuda()
            if len(image.size()) == 4:
                bs, c, h, w = image.size()
                n_crops = 1
            else:
                bs, n_crops, c, h, w = image.size()
            output = torch.softmax(model(image.view(-1, c, h, w)), dim=1)
            pred = output.view(bs, n_crops, -1).mean(1).argmax(1).cpu()
            for i in range(len(pred)):
                writer.writerow([model_spec['model_name'] + '_' + filenames[i + idx], pred[i].item()])
            idx += len(pred)

    print ('[%s] done, please use concat.py to concat the prediction of another model before submit' % (datetime.datetime.now().ctime()))
