import math
import sys
import random
import time
import datetime
from typing import Iterable
import torch
import torchvision
import torch.nn.functional as Func
import PIL
import numpy as np
import util.misc as utils
import torch.nn.functional as F

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def augment(x, l, device, beta=0.5):
    mixs = []
    try:
        x=x.tensors
    except:
        pass
    mix = torch.distributions.beta.Beta(beta, beta).sample([x.shape[0], 1, 1, 1])
    mix = torch.maximum(mix, 1 - mix)
    mix = mix.to(device)
    mixs.append(mix)
    xmix = x * mix + torch.flip(x,(0,)) * (1 - mix)
    lmix = l * mix + torch.flip(l,(0,)) * (1 - mix)
    return xmix, lmix, mixs

def mix_targets(samples, targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    aug_samples, aug_targets, rates = augment(samples, target_masks, device)
    return aug_samples, aug_targets, rates

def convert_targets(targets):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def convert_targets_values(targets):
    indicator = torch.zeros((len(targets),4,212,212)).cuda()
    for index in range(len(targets)):
        t = targets[index]
        values = t["lab_values"]
        values = [i for i in values if i <4]
        for value in values:
            indicator[index,value,:,:] += 1
    return indicator

def Cutout_augment(x, l, device, beta=1):
    lams = []
    try:
        x=x.tensors
    except:
        pass
    lam = torch.distributions.beta.Beta(beta, beta).sample([x.shape[0], 1, 1, 1])
    bboxs = []
    x_flip = torch.flip(x,(0,))
    l_flip = torch.flip(l,(0,))
    for index in range(x.shape[0]):
        bbx1, bby1, bbx2, bby2= rand_bbox(x.shape, lam[index,0,0,0])
        x[index,:,bbx1:bbx2,bby1:bby2] = 0
        l[index,:,bbx1:bbx2,bby1:bby2]= 0
        bboxs.append([bbx1, bby1, bbx2, bby2])
    return x, l, bboxs

def Cutout_targets(samples, targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    aug_samples, aug_targets, bboxs = Cutout_augment(samples, target_masks, device)
    return aug_samples, aug_targets, bboxs

def to_onehot_dim4(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot_dim5(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def rotate(imgs,labels):
    num = imgs.shape[0]
    imgs_out_list = []
    labels_out_list = []
    angles = []
    
    for i in range(num):
        img = imgs[i,:,:,:]
        label = labels[i,:,:,:]

        angle = float(torch.empty(1).uniform_(0.0, 360.0).item())
        
        rotated_img = torchvision.transforms.functional.rotate(img, angle, PIL.Image.NEAREST, False, None)
        rotated_label = torchvision.transforms.functional.rotate(label, angle, PIL.Image.NEAREST, False, None)
        
        imgs_out_list.append(rotated_img)
        labels_out_list.append(rotated_label)
        
        angles.append(angle)
    
    imgs_out = torch.stack(imgs_out_list)
    labels_out = torch.stack(labels_out_list)
    return imgs_out, labels_out, angles

def flip(imgs, labels):
    imgs_list = []
    labels_list = []
    flips = []
    for i in range(imgs.shape[0]):
        img = imgs[i,:,:,:]
        label = labels[i,:,:,:]

        flipped_img = img
        flipped_label = label

        flip_choice = int(random.random()*4)
        if flip_choice == 0:
            pass

        if flip_choice == 1:
            flipped_img = torch.flip(flipped_img,[1])
            flipped_label = torch.flip(flipped_label,[1])

        if flip_choice == 2:
            flipped_img = torch.flip(flipped_img,[2])
            flipped_label = torch.flip(flipped_label,[2])

        if flip_choice == 3:
            flipped_img = torch.flip(flipped_img,[1,2])
            flipped_label = torch.flip(flipped_label,[1,2])

        flips.append(flip_choice)
        imgs_list.append(flipped_img)
        labels_list.append(flipped_label)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    return imgs_out, labels_out, flips

def flip_back(outputs, flips):
    outs = []
    for i in range(outputs["pred_masks"].shape[0]):
        output = outputs["pred_masks"][i,:,:,:]
        flip_choice = flips[i]
        flipped_img = output

        if flip_choice == 0:
            pass

        if flip_choice == 1:
            flipped_img = torch.flip(flipped_img,[1])

        if flip_choice == 2:
            flipped_img = torch.flip(flipped_img,[2])

        if flip_choice == 3:
            flipped_img = torch.flip(flipped_img,[1,2])

        outs.append(flipped_img)
        outs = torch.stack(outs)
        return {"pred_masks":outs}

def rotate_back(outputs,angles):
    num = outputs["pred_masks"].shape[0]
    outputs_out_list = []
    
    for i in range(num):
        output = outputs["pred_masks"][i,:,:,:]
        angle = -angles[i]
        
        rotated_output =  torchvision.transforms.functional.rotate(output, angle, PIL.Image.NEAREST, False, None)
        
        outputs_out_list.append(rotated_output)
    
    outputs_out = torch.stack(outputs_out_list) 
    return {"pred_masks":outputs_out}

def Cutout(imgs,labels, device, n_holes=1, length=32):
    labels = [t["masks"] for t in labels]
    labels = torch.stack(labels)

    h = imgs.shape[2]
    w = imgs.shape[3]
    num = imgs.shape[0]
    labels_list = []
    imgs_list = []
    masks_list = []

    for i in range(num):
        label = labels[i,:,:,:]
        img = imgs[i,:,:,:]
        mask = np.ones((1, h, w), np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[0, y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask
        label = label * mask
        imgs_list.append(img)
        labels_list.append(label)
        masks_list.append(mask)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    masks_out = torch.stack(masks_list)

    return imgs_out, labels_out, masks_out

@torch.no_grad()
def sinkhorn(out, epsilon=0.05, iterations = 3):
    Q = torch.exp(out/epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def train_one_epoch(backbone, head, criterion, dataloader,optimizer, device, epoch):
    backbone.train()
    head.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    numbers = len(dataloader)
    iterats = iter(dataloader)
    total_steps = numbers
    start_time = time.time()

    for step in range(total_steps):
        start = time.time()
        samples, targets = next(iterats)
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]
        targets = convert_targets(targets)
        samples.tensors = samples.tensors.float()
        outputs = head(backbone(samples.tensors))
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in ['loss_CrossEntropy',"loss_multiDice"])
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items() if k in ['loss_CrossEntropy',"loss_multiDice"]}

        final_losses = losses

        optimizer.zero_grad()
        final_losses.backward()
        optimizer.step()

        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
    # gather the stats from all processes  
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats


@torch.no_grad()
def evaluate(backbone, head, criterion, dataloader, device):
    backbone.eval()
    head.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('loss_multiDice', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    print_freq = 10
    numbers = len(dataloader)
    iterats = iter(dataloader)
    total_steps = numbers
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    for step in range(total_steps):
        start = time.time()
        samples, targets = next(iterats)
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        targets_onehot= convert_targets(targets)
        outputs = head(backbone(samples.tensors))

        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict.keys()}
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        # if step % round(total_steps / 7.) == 0:
        #     ##original
        #     sample_list.append(samples.tensors[0])
        #     ##
        #     _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
        #     output_list.append(pre_masks)
        #
        #     ##original
        #     target_list.append(targets[0]['masks'])
        #     ##
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    # writer.add_scalar('avg_loss', stats['loss_CrossEntropy'], epoch)
    # visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)
    return stats