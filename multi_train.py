import os
import util.misc as utils
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import torch
import torch.nn.functional as Func
import torchvision.transforms.functional as F
from tensorboardX import SummaryWriter
from validation import CrossModalSegNetValidation
from criterion.loss import SegmentationLoss
import time
import numpy as np
import PIL
import copy


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

def rotate(img, mask):
    angle = np.random.uniform(0,360)
    rotated_img = F.rotate(img, angle, PIL.Image.NEAREST)
    rotated_mask = F.rotate(mask, angle, PIL.Image.NEAREST)
    return rotated_img, rotated_mask, angle

def rotate_angle(img, angle):
    rotated_img = F.rotate(img, angle, PIL.Image.NEAREST)
    return rotated_img

def multi_train(args1, args, backbone1, backbone3, head1, head3, Train_Image, Train_loader, Valid_Image, Valid_loader, dataloader_MSCMR, optimizer, criterion, device, epoch):
    seg_loss = SegmentationLoss().cuda()
    writer = SummaryWriter()
    best_dice = 0
    backbone1.train()
    head1.train()
    backbone3.train()
    head3.train()
    criterion.train()

    iterats_MSCMR = iter(dataloader_MSCMR)

    train_C0 = torch.FloatTensor(args1.batch_size, 1, args1.dim, args1.dim).cuda()
    train_DE = torch.FloatTensor(args1.batch_size, 1, args1.dim, args1.dim).cuda()
    train_T2 = torch.FloatTensor(args1.batch_size, 1, args1.dim, args1.dim).cuda()
    train_label = torch.FloatTensor(args1.batch_size, 5, args1.dim, args1.dim).cuda()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    IterCount = int(len(Train_Image) / args1.batch_size)

    for iteration in range(IterCount):
        start = time.time()

        names = []
        for name, _ in backbone1.named_parameters():
            if "Up" not in name:
                names.append(name)
        selected_index = random.randint(1, len(names)/2)

        img_C0, img_DE, img_T2, label, _ = next(Train_loader)

        train_C0.copy_(img_C0)
        train_DE.copy_(img_DE)
        train_T2.copy_(img_T2)
        train_label.copy_(label)

        input = torch.cat([train_C0, train_DE, train_T2],1)
        out = head1(backbone1(input))
        seg = out["seg"]
        loss1 = seg_loss(seg, train_label)

        B = train_C0.shape[0]
        C = train_C0.shape[1]
        beta = torch.tensor(np.random.beta(0.5, 0.5, [B,1,1,1])).cuda()
        train_C0_mix = (beta*train_C0 + (1-beta)*torch.flip(train_C0,[0])).float()
        train_DE_mix = (beta*train_DE + (1-beta)*torch.flip(train_DE,[0])).float()
        train_T2_mix = (beta*train_T2 + (1-beta)*torch.flip(train_T2,[0])).float()
        y_labeled_mixed = (beta*train_label + (1-beta)*torch.flip(train_label,[0])).float()
        input_mix = torch.cat([train_C0_mix, train_DE_mix, train_T2_mix],1)
        out_mix = head1(backbone1(input_mix))
        seg_mix = out_mix["seg"]
        pred_mix = beta*seg+ (1-beta)*torch.flip(seg,[0])
        ce_mixed_loss1 = seg_loss(seg_mix, y_labeled_mixed, False)
        flat_mix_pred = seg_mix.reshape(B,C,-1)
        flat_pred_mix = pred_mix.reshape(B,C,-1)
        mix_consistency1 = 1 - Func.cosine_similarity(flat_pred_mix,flat_mix_pred,dim=2).mean()


        rotated_input1, rotated_train_label, angle1 = rotate(input, train_label)
        
        backbone_mix = copy.deepcopy(backbone1)
        model_mix_ratio = np.random.beta(0.5, 0.5)
        for p_out, p_in1, p_in2 in zip(backbone_mix.named_parameters(), backbone1.parameters(), backbone3.parameters()):
            if p_out[0] == names[selected_index*2-2] or p_out[0] == names[selected_index*2-1]:
                p_out[1].data = torch.nn.Parameter(p_in1 * model_mix_ratio + p_in2 * (1 - model_mix_ratio))
        
        out_mix13 = head1(backbone_mix(rotated_input1))
        seg_mix13 = out_mix13["seg"]
        seg_mix_back13 = rotate_angle(seg_mix13, -angle1)
        dx = seg.shape[0]
        invariant_loss_mix13 = - Func.cosine_similarity(seg.reshape((B,C, -1)), seg_mix_back13.reshape((B,C, -1)), dim=1).mean()
        loss1_mix13 = seg_loss(seg_mix13, rotated_train_label)

        try:
            samples_MSCMR, targets_MSCMR = next(iterats_MSCMR)
        except:
            iterats_MSCMR = iter(dataloader_MSCMR)
            samples_MSCMR, targets_MSCMR = next(iterats_MSCMR)

        samples_MSCMR = samples_MSCMR.to(device)
        targets_MSCMR = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets_MSCMR]
        targets_MSCMR = convert_targets(targets_MSCMR)
        samples_MSCMR.tensors = samples_MSCMR.tensors.float()

        outputs_MSCMR = head3(backbone3(samples_MSCMR.tensors))
        loss_dict_MSCMR = criterion(outputs_MSCMR, targets_MSCMR)
        
        B = samples_MSCMR.tensors.shape[0]
        C = 4
        beta = torch.tensor(np.random.beta(0.5, 0.5, [B,1,1,1])).cuda()
        samples_MSCMR_mix = (beta*samples_MSCMR.tensors + (1-beta)*torch.flip(samples_MSCMR.tensors,[0])).float()
        targets_MSCMR_mix = (beta*targets_MSCMR + (1-beta)*torch.flip(targets_MSCMR,[0])).float()
        outputs_MSCMR_mix = head3(backbone3(samples_MSCMR_mix))
        pred_mix = beta*outputs_MSCMR["seg"]+ (1-beta)*torch.flip(outputs_MSCMR["seg"],[0])
        ce_MSCMR = -targets_MSCMR_mix[:,0:4,:,:]*torch.log(outputs_MSCMR_mix["seg"]+1e-5)
        # ce_mixed_loss3 = ce_MSCMR.sum()/(ce_MSCMR.sum()+1e-5)
        ce_mixed_loss3 = ce_MSCMR.sum()/(targets_MSCMR_mix[:,0:4,:,:].sum()+1e-5)
        flat_mix_pred = outputs_MSCMR_mix["seg"].reshape(B,C,-1)
        flat_pred_mix = pred_mix.reshape(B,C,-1)
        mix_consistency3 = 1 - Func.cosine_similarity(flat_pred_mix,flat_mix_pred,dim=2).mean()


        rotated_samples_MSCMR, rotated_targets_MSCMR, angle3 = rotate(samples_MSCMR.tensors, targets_MSCMR)

        backbone_mix = copy.deepcopy(backbone3)
        model_mix_ratio = np.random.beta(0.5, 0.5)
        for p_out, p_in1, p_in2 in zip(backbone_mix.named_parameters(), backbone3.parameters(), backbone1.parameters()):
            if p_out[0] == names[selected_index*2-2] or p_out[0] == names[selected_index*2-1]:
                p_out[1].data = torch.nn.Parameter(p_in1 * model_mix_ratio + p_in2 * (1 - model_mix_ratio))
                
        outputs_mix_MSCMR_13 = head3(backbone_mix(rotated_samples_MSCMR))
        seg_mix_back_MSCMR_13 = rotate_angle(outputs_mix_MSCMR_13["seg"], -angle3)
        invariant_loss_2_mix_MSCMR_13 = - Func.cosine_similarity(outputs_MSCMR["seg"].reshape((B,C, -1)), seg_mix_back_MSCMR_13.reshape((B,C, -1)), dim=2).mean()
        loss_dict_mix_MSCMR_13 = criterion(outputs_mix_MSCMR_13, rotated_targets_MSCMR)
        
        
        weight_dict = criterion.weight_dict
        loss2_MSCMR = sum(
            loss_dict_MSCMR[k] * weight_dict[k] for k in loss_dict_MSCMR.keys() if k in ['loss_CrossEntropy', "loss_multiDice"])
        loss2_mix_MSCMR_13 = sum(
            loss_dict_mix_MSCMR_13[k] * weight_dict[k] for k in loss_dict_mix_MSCMR_13.keys() if k in ['loss_CrossEntropy', "loss_multiDice"])
        
        optimizer.zero_grad()
        loss1_sup = (loss1 + ce_mixed_loss1 + mix_consistency1) 
        loss1_unsup = loss1_mix13 + invariant_loss_mix13
        if iteration == 0:
            print("myops:", "loss1:", loss1.item(), "loss1 mix:", loss1_mix13.item(), "invariant mix:", invariant_loss_mix13.item())#"invariant end:", invariant_loss_1_end.item()
        loss3_sup = loss2_MSCMR + ce_mixed_loss3 + mix_consistency3 
        loss3_unsup = loss2_mix_MSCMR_13 + invariant_loss_2_mix_MSCMR_13
        if iteration == 0:
            print("MSCMR:", "loss3:", loss2_MSCMR.item(),"loss3 mix:", loss2_mix_MSCMR_13.item(), "invariant mix MSCMR:", invariant_loss_2_mix_MSCMR_13.item())#"invariant end:", invariant_loss_2_end.item()
        (loss1_sup+loss3_sup+loss1_unsup+loss3_unsup).backward()
        optimizer.step()

        loss_dict_reduced = utils.reduce_dict(loss_dict_MSCMR)
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items() if
                                    k in ['loss_CrossEntropy', "loss_multiDice"]}

        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        itertime = time.time() - start
        datatime = time.time() - start
        metric_logger.log_every(iteration, IterCount, datatime, itertime, print_freq, header)

        with open('log_training.txt', 'a') as segment_log:
            segment_log.write("==> Epoch: {:0>3d}/{:0>3d} - ".format(epoch + 1, args1.end_epoch))
            segment_log.write("Iteration: {:0>3d}/{:0>3d} - ".format(iteration + 1, IterCount))
            segment_log.write("LR: {:.6f} - ".format(float(optimizer.param_groups[0]['lr'])))
            segment_log.write("loss: {:.6f}\n".format(loss1.detach().cpu()))

        # write to tensorboard
        writer.add_scalar('seg loss', loss1.detach().cpu(), epoch * (IterCount + 1) + iteration)

    # Validation
    backbone1.eval()
    head1.eval()
    avg_dice_valid = CrossModalSegNetValidation(args1, epoch, backbone1, head1, Valid_Image, Valid_loader, writer,
                                                'result_validation.txt', 'valid scar', 'valid edema')
    writer.close()
    if avg_dice_valid > best_dice or avg_dice_valid > 0.5:
        if avg_dice_valid > best_dice:
            best_dice = avg_dice_valid
        torch.save(backbone1.state_dict(), os.path.join('/data/kzhang/Modelmix_MSCMR_MyoPS_5shot/MyoPS',
                                                     str(avg_dice_valid.cpu().numpy()) + '_' + str(
                                                         epoch + 1) + 'backbone.pth'))
        torch.save(head1.state_dict(), os.path.join('/data/kzhang/Modelmix_MSCMR_MyoPS_5shot/MyoPS',
                                                     str(avg_dice_valid.cpu().numpy()) + '_' + str(
                                                         epoch + 1) + 'head_myops.pth'))

