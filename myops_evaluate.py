import nibabel as nib
import numpy as np
import os
from medpy.metric.binary import hd
from medpy.metric.binary import dc
import pandas as pd
# gd_dir = "D:/MyoPS_DE_scribble/val_gd/"
gd_dir = "/data/MyoPS_test/myops2020_test20_gd_notforpublic/"
# pred_dir = "C:/Users/19272/Desktop/MyoPS_test/myops_test/val_5/test_1_val/"
# pred_dir = "/data/kzhang99/myops_sensitivy/test_15_v4/test_9" # test_19 0.58
# output_folder = "D:/Weak_supervised/ZScribbleNet/SOTA/MyoPS/tables/"
pred_dir = "/home/kzhang99/ModelMix/Modelmix_cross_datasets/test/test_6/" # test_19 0.58

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine, header)
    if os.path.exists(img_path):
        os.remove(img_path)
        nimg.to_filename(img_path)
    else:
        nimg.to_filename(img_path)

gd_files = sorted(os.listdir(gd_dir))
pred_files = sorted([i for i in os.listdir(pred_dir) if "PP" not in i])

file_names = []
structure_names = []
dices_list = []
hd_list = []
structures_dict = {1220: 'Edema', 2221: 'Scar'}
count = 0

for gd_file, pred_file in list(zip(gd_files, pred_files)):
    gd_path = os.path.join(gd_dir, gd_file)
    pred_path = os.path.join(pred_dir, pred_file)
    gt, gd_affine, gd_header = load_nii(gd_path)
    pred, pred_affine, pred_header = load_nii(pred_path)
    pred = np.round(pred)
    gd = np.round(gt)
    for struc in [1220, 2221]:
        gt_binary = (gt == struc) * 1
        pred_binary = (pred == struc) * 1
        if struc == 1220:
            gt_binary = gt_binary + (gt == 2221) * 1
            pred_binary = pred_binary + (pred == 2221) * 1
        if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
            dices_list.append(1)
        elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(gt_binary) > 0:
            dices_list.append(0)
            count += 1
        else:
            dices_list.append(dc(gt_binary, pred_binary))

        file_names.append(gd_file)
        structure_names.append(structures_dict[struc])
df = pd.DataFrame({'dice': dices_list, 'struc': structure_names, 'filename': file_names})
print("Scar:", df[df['struc']=='Scar']['dice'].mean(),df[df['struc']=='Scar']['dice'].std())
print("Edema:", df[df['struc']=='Edema']['dice'].mean(),df[df['struc']=='Edema']['dice'].std())
print("Avg:", df['dice'].mean(), df['dice'].std())
csv_path = "5_UNet_full.csv"
df.to_csv(csv_path)



