import numpy as np
import nibabel as nib
import torch
import os
from medpy.metric.binary import dc
from medpy.metric.binary import hd95
from medpy.metric.binary import hd
import pandas as pd
import glob
import re
import shutil
import copy
from skimage import measure
from scipy.ndimage import zoom
import util.misc as utils


def makefolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


@torch.no_grad()
def infer(backbone, head, criterion, device):
    backbone.eval()
    head.eval()
    criterion.eval()

    test_folder = "/home/kzhang99/ModelMix/scribble/val/images/"
    label_folder = "/home/kzhang99/ModelMix/scribble/val/labels/"
    output_folder = "/home/kzhang99/ModelMix/scribble/self_tmp/"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    makefolder(output_folder)

    test_files = sorted(os.listdir(test_folder))
    label_files = sorted(os.listdir(label_folder))
    assert len(test_files) == len(label_files)

    for file_index in range(len(test_files)):
        test_file = test_files[file_index]
        label_file = label_files[file_index]
        file_mask = os.path.join(label_folder, label_file)
        mask_dat = load_nii(file_mask)

        img_path = os.path.join(test_folder, test_file)
        img_dat = load_nii(img_path)
        img = img_dat[0].copy()
        img = img.astype(np.float32)
        img = np.divide((img - np.mean(img)), np.std(img))

        predictions = []
        print("image shape:", img.shape)
        for slice_index in range(img.shape[2]):
            img_slice = img[:, :, slice_index]
            nx = 256
            ny = 256
            x, y = img_slice.shape
            # x_s = (x - nx) // 2
            # y_s = (y - ny) // 2
            #
            # slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
            slice_cropped = zoom(img_slice, (nx / x, ny / y))
            img_slice = np.divide((slice_cropped - np.mean(slice_cropped)), np.std(slice_cropped))
            img_slice = np.reshape(img_slice, (1, 1, nx, ny))
            img_slice = np.repeat(img_slice, 3, 1)
            img_slice = torch.from_numpy(img_slice)
            img_slice = img_slice.to(device)
            img_slice = img_slice.float()
            outputs = head(backbone(img_slice))
            softmax_out = outputs["seg"]
            softmax_out = softmax_out.detach().cpu().numpy()

            prediction_cropped = np.squeeze(softmax_out[0, ...])

            slice_predictions = np.zeros((4, x, y))
            # slice_predictions[:, x_s:x_s + nx, y_s:y_s + ny] = prediction_cropped
            slice_predictions = zoom(prediction_cropped, (1, x / nx, y / ny))
            prediction = np.uint8(np.argmax(slice_predictions, axis=0))
            predictions.append(prediction)

        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1, 2, 0))
        dir_pred = os.path.join(output_folder, "predictions")
        makefolder(dir_pred)
        out_file_name = os.path.join(dir_pred, label_file)
        out_affine = mask_dat[1]
        out_header = mask_dat[2]

        save_nii(out_file_name, prediction_arr, out_affine, out_header)

        dir_gt = os.path.join(output_folder, "masks")
        makefolder(dir_gt)
        mask_file_name = os.path.join(dir_gt, label_file)
        save_nii(mask_file_name, mask_dat[0], out_affine, out_header)

    filenames_gt = sorted(glob.glob(os.path.join(dir_gt, '*')), key=natural_order)
    filenames_pred = sorted(glob.glob(os.path.join(dir_pred, '*')), key=natural_order)
    file_names = []
    structure_names = []
    dices_list = []
    structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}
    count = 0
    avg_dice = 0
    num_slices = 0
    for p_gt, p_pred in zip(filenames_gt, filenames_pred):
        print(p_gt, p_pred)
        gt, _f, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        gt = np.round(gt)
        pred = np.round(pred)
        for struc in [3, 1, 2]:
            gt_binary = (gt == struc) * 1
            pred_binary = (pred == struc) * 1
            if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                dices_list.append(1)
            elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(
                    gt_binary) > 0:
                dices_list.append(0)
                count += 1
            else:
                dices_list.append(dc(gt_binary, pred_binary))
            file_names.append(os.path.basename(p_pred))
            structure_names.append(structures_dict[struc])

        for index in range(gt.shape[-1]):
            inter = 2 * pred[:, :, index] * gt[:, :, index] + 1e-10
            denom = pred[:, :, index].sum() + gt[:, :, index].sum() + 1e-10
            dice = inter.sum() / denom
            avg_dice += dice
            num_slices += 1
    avg_dice = avg_dice / num_slices
    print(avg_dice)

    df = pd.DataFrame({'dice': dices_list, 'struc': structure_names, 'filename': file_names})
    print(df[df['struc'] == 'LV']['dice'].mean(), df[df['struc'] == 'LV']['dice'].std())
    print(df[df['struc'] == 'Myo']['dice'].mean(), df[df['struc'] == 'Myo']['dice'].std())
    print(df[df['struc'] == 'RV']['dice'].mean(), df[df['struc'] == 'RV']['dice'].std())
    print(df['dice'].mean(), df['dice'].std())
