import os
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from itertools import cycle
from torch.utils.data import DataLoader
from utils.tools import weights_init
from validation import CrossModalSegNetValidation
from utils.dataloader import CrossModalDataLoader
from criterion.loss import SegmentationLoss
import torch.nn.functional as F
import time
import numpy as np
import datetime


def convert_lab_values(lab_values):
    print(torch.unique(lab_values))
    lab_values = lab_values.long()
    indicator = torch.zeros((lab_values.shape[0], 5, 192, 192)).cuda()
    for index in range(lab_values.shape[0]):
        values = lab_values[index]
        values = [i for i in values if i < 5]
        for value in values:
            indicator[index, value, :, :] += 1
    return indicator


def CrossModalSegNetTrain(args, backbone, head, Train_Image, Train_loader, Valid_Image, Valid_loader, epoch, optimizer):
    # # loss definitions
    seg_loss = SegmentationLoss().cuda()
    #
    # # optimizer
    # optimizer = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    # if not os.path.exists('checkpoints'):
    #     os.makedirs('checkpoints')
    #
    # # initialize summary writer
    writer = SummaryWriter()
    best_dice = 0
    # for epoch in range(args.start_epoch, args.end_epoch):
    backbone.train()
    head.train()

    start_time = time.time()

    train_C0 = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
    train_DE = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
    train_T2 = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
    train_label = torch.FloatTensor(args.batch_size, 5, args.dim, args.dim).cuda()

    IterCount = int(len(Train_Image) / args.batch_size)

    for iteration in range(IterCount):
        # Sup
        img_C0, img_DE, img_T2, label, indicator = next(Train_loader)
        # indicator = indicator.cuda()

        train_C0.copy_(img_C0)
        train_DE.copy_(img_DE)
        train_T2.copy_(img_T2)
        train_label.copy_(label)
        input = torch.cat([train_C0, train_DE, train_T2],1)
        out = head(backbone(input))
        seg = out["seg"]

        # seg loss
        loss = seg_loss(seg, train_label)

        B = seg.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # write to log
        with open('log_training.txt', 'a') as segment_log:
            segment_log.write("==> Epoch: {:0>3d}/{:0>3d} - ".format(epoch + 1, args.end_epoch))
            segment_log.write("Iteration: {:0>3d}/{:0>3d} - ".format(iteration + 1, IterCount))
            segment_log.write("LR: {:.6f} - ".format(float(optimizer.param_groups[0]['lr'])))
            segment_log.write("loss: {:.6f}\n".format(loss.detach().cpu()))

        # write to tensorboard
        writer.add_scalar('seg loss', loss.detach().cpu(), epoch * (IterCount + 1) + iteration)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Validation
    backbone.eval()
    head.eval()
    avg_dice_valid = CrossModalSegNetValidation(args, epoch, backbone, head, Valid_Image, Valid_loader, writer,
                                                'result_validation.txt', 'valid scar', 'valid edema')
    if avg_dice_valid > best_dice or avg_dice_valid > 0.5:
        if avg_dice_valid > best_dice:
            best_dice = avg_dice_valid
        torch.save(backbone.state_dict(), os.path.join('/data/ModelMix_results/modelmix_myops/',
                                                     str(avg_dice_valid.cpu().numpy()) + '_' + str(
                                                         epoch + 1) + 'backbone.pth'))
        torch.save(head.state_dict(), os.path.join('/data/ModelMix_results/modelmix_myops/',
                                                     str(avg_dice_valid.cpu().numpy()) + '_' + str(
                                                         epoch + 1) + 'head.pth'))
