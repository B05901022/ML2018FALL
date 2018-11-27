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

def evaluate(model, mode, eval_spec):
    assert mode in ['valid']

    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        accs = 0.0
        losses = 0.0
        total = 0
        for batch in loader.get_dataloader(mode, eval_spec["transform"], int(eval_spec["batchsize"])):
            image = batch['image'].cuda()
            if len(image.size()) == 4:
                bs, c, h, w = image.size()
                n_crops = 1
            else:
                bs, n_crops, c, h, w = image.size()
            output = torch.softmax(model(image.view(-1, c, h, w)), dim=1)
            loss = loss_fn(output.cpu(), batch['label'].repeat(n_crops).view(n_crops, -1).transpose(0, 1).reshape(-1))
            losses += loss.item() * bs
            pred = output.view(bs, n_crops, -1).mean(1).argmax(1).cpu()
            acc = pred.eq(batch['label']).float().sum()
            accs += acc.item()
            total += bs
    return accs / total, losses / total

if __name__ == '__main__':
    model_spec = train.get_model_spec()
    seed = train.get_random_seed()
    if seed is not None:
        assert isinstance(seed, int)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    assert "model_name" in model_spec and "pretrained" in model_spec
    assert isinstance(model_spec["pretrained"], bool)
    assert isinstance(model_spec["model_name"], str)
    model = m.get_model(model_spec["model_name"], model_spec["pretrained"]).cuda()

    optimizer = train.get_optimizer(model.parameters())

    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    eval_spec = train.get_eval_spec()
    assert "transform" in eval_spec and "batchsize" in eval_spec
    assert int(eval_spec["batchsize"]) > 0 and int(eval_spec["batchsize"]) <= 64

    train_history = []
    validation_history = []
    print ('[%s] training start' % (datetime.datetime.now().ctime()))
    for i in range(50):
        model.train()
        train_history.append([])
        epoch_spec = train.before_epoch(copy.deepcopy(train_history), copy.deepcopy(validation_history))
        assert "transform" in epoch_spec and "batchsize" in epoch_spec
        assert int(epoch_spec["batchsize"]) > 0 and int(epoch_spec["batchsize"]) <= 64
        assert epoch_spec["transform"] is not None

        losses = 0.0
        accs = 0.0
        n = 0
        for batch in loader.get_dataloader('train', epoch_spec["transform"], int(epoch_spec["batchsize"])):
            batch_spec = train.before_batch(copy.deepcopy(train_history), copy.deepcopy(validation_history))
            assert "optimizer" in batch_spec and "batch_norm" in batch_spec
            if model_spec["model_name"] == "mobilenetv2":
                assert "drop_out" in batch_spec

            if batch_spec["optimizer"] is not None:
                assert isinstance(batch_spec["optimizer"], dict)
                for group in optimizer.param_groups:
                    for k in batch_spec["optimizer"]:
                        if k in group:
                            group[k] = batch_spec["optimizer"][k]

            if batch_spec["batch_norm"] != None:
                assert float(batch_spec["batch_norm"]) >= 0 and float(batch_spec["batch_norm"]) <= 1
                for layer in model.modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        layer.momentum = float(batch_spec["batch_norm"])

            if "drop_out" in batch_spec and batch_spec["drop_out"] is not None:
                assert float(batch_spec["drop_out"]) >= 0. and float(batch_spec["drop_out"]) <= 1.
                model.set_dropout(float(batch_spec["drop_out"]))

            optimizer.zero_grad()
            output = model(batch['image'].cuda())
            pred = output.cpu().argmax(1)
            acc = pred.eq(batch['label']).float().mean()
            accs += acc.item()

            loss = loss_fn(output, batch['label'].cuda())
            loss.backward()
            losses += loss.item()

            train_history[-1].append([acc.item(), loss.item()])
            n += 1
            optimizer.step()

        print ('[%s] epoch %d: %.4f %.4f' % (datetime.datetime.now().ctime(), i + 1, losses / n, accs / n))
        acc, loss = evaluate(model, 'valid', eval_spec)
        validation_history.append([acc, loss])
        print ('[%s] valid : %.4f %.4f' % (datetime.datetime.now().ctime(), loss, acc))

        filename = train.save_model_as(copy.deepcopy(train_history), copy.deepcopy(validation_history))
        if filename is not None:
            assert isinstance(filename, str)
            torch.save(model.state_dict(), filename)
