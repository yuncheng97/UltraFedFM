# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import sys
import math
import torch
import logging
import numpy as np
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from pycm import *
from timm.data import Mixup
from timm.utils import accuracy
from torchvision import transforms
from PIL import Image, ImageFilter
from typing import Iterable, Optional


import util.misc as misc
import util.lr_sched as lr_sched
import util.metrics as metrics


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, logging, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    test_acc = torchmetrics.Accuracy('multiclass', average='micro', num_classes=args.nb_classes).cuda()
    test_recall = torchmetrics.Recall('multiclass', average='macro', num_classes=args.nb_classes).cuda()
    test_precision = torchmetrics.Precision('multiclass', average='macro', num_classes=args.nb_classes).cuda()
    test_f1 = torchmetrics.F1Score('multiclass', average='macro', num_classes=args.nb_classes).cuda()
    test_auroc = torchmetrics.AUROC("multiclass", average='macro', num_classes=args.nb_classes).cuda()
    test_aupr = torchmetrics.AveragePrecision("multiclass", average='macro', num_classes=args.nb_classes).cuda()
    test_spe = torchmetrics.Specificity('multiclass', average='macro', num_classes=args.nb_classes).cuda()
    test_cm = torchmetrics.ConfusionMatrix('multiclass', num_classes=args.nb_classes).cuda()
    total_loss = 0.
    count = 0

    prediction_decode_list = []
    true_label_decode_list = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=args.nb_classes)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

            prediction_softmax = nn.Softmax(dim=1)(output)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        count += batch_size

        test_acc(output.argmax(1), target)
        test_recall(output.argmax(1), target)
        test_precision(output.argmax(1), target)
        test_f1(output.argmax(1), target)
        test_spe(output.argmax(1), target)
        test_auroc(output, target)
        test_aupr(output, target)
        test_cm(output, target)

    cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
    if args.nb_classes > 2:
        total_acc = test_acc.compute()
    else:
        total_acc = cm.ACC_Macro
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auroc = test_auroc.compute()
    total_aupr = test_aupr.compute()
    total_f1 = test_f1.compute()
    total_spe = test_spe.compute()
    total_cm = test_cm.compute()
    total_loss = total_loss / count
    
    print('TEST Epoch:{epoch} * ACC {acc:.3f} Precision {prec:.3f} Recall {rec:.3f} F1 {f1:.3f} AUROC {auroc:.3f} AUPR {aupr:.3f} SPE {spe:.3f} Loss {loss:.3f} \n Confusion Matrix \n {cm}'
        .format(epoch=epoch, acc=100*total_acc, prec=total_precision, rec=total_recall, f1=total_f1, auroc=total_auroc, aupr=total_aupr, spe=total_spe, loss=total_loss, cm=total_cm))
    logging.info('TEST Epoch:{epoch}* ACC {acc:.3f} Precision {prec:.3f} Recall {rec:.3f} F1 {f1:.3f} AUROC {auroc:.3f} AUPR {aupr:.3f} SPE {spe:.3f} Loss {loss:.3f} \n Confusion Matrix \n {cm}'
        .format(epoch=epoch, acc=100*total_acc, prec=total_precision, rec=total_recall, f1=total_f1, auroc=total_auroc, aupr=total_aupr, spe=total_spe, loss=total_loss, cm=total_cm))

    return {'acc': 100*total_acc, 'precision': total_precision, 'recall': total_recall, 'f1': total_f1, 'auroc': total_auroc, 'aupr': total_aupr, 'spe': total_spe, 'loss': total_loss, 'cm': total_cm, 'y_true': true_label_decode_list, 'y_pred': prediction_decode_list}