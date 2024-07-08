import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import argparse
# from process import LargestConnectedComponents
from utils.transform import Truncate_and_Normalize, ImageTransform, ResultTransform, \
    Zero_Mean_Unit_Variance_Normalization
from models import build_model
# from network.model import CrossModalSegNet
from utils.config import config
from main import get_args_parser
import warnings

warnings.filterwarnings("ignore")


def predict(args, args_myops, backbone_path, head_path, epoch):
    # model definition
    # model = CrossModalSegNet(in_chs=3, out_chs=5)
    backbone1, Head1, Head2, criterion, visualizer = build_model(args)
    device = torch.device(args.device)
    backbone1.to(device)
    Head1.to(device)

    backbone1.load_state_dict(torch.load(backbone_path, map_location='cpu'))
    Head1.load_state_dict(torch.load(head_path, map_location='cpu'))

    backbone1.eval()
    Head1.eval()

    if not os.path.exists('test/test_' + str(epoch)):
        os.makedirs('test/test_' + str(epoch))

    test_img = pd.read_csv(args_myops.path + 'MyoPS/test.csv')

    normalize = Truncate_and_Normalize()
    slice_normalize = Zero_Mean_Unit_Variance_Normalization()
    image_transform = ImageTransform(args_myops.dim, 'Test')
    result_transform = ResultTransform()

    for i in range(int(len(test_img))):

        prefix_data = os.path.join(args_myops.path + 'MyoPS/' + test_img.iloc[i]["stage"], test_img.iloc[i]["file_name"])
        dim_x, dim_y, dim_z = test_img.iloc[i]["dx"], test_img.iloc[i]["dy"], test_img.iloc[i]["dz"]
        res = torch.zeros([dim_z, dim_x, dim_y])

        # get data [x,y,z]
        C0_raw = nib.load(prefix_data + '_C0.nii.gz')
        DE_raw = nib.load(prefix_data + '_DE.nii.gz')
        T2_raw = nib.load(prefix_data + '_T2.nii.gz')
        img_affine = C0_raw.affine

        C0_slice = normalize(C0_raw.get_fdata()).astype(np.float32)
        DE_slice = normalize(DE_raw.get_fdata()).astype(np.float32)
        T2_slice = normalize(T2_raw.get_fdata()).astype(np.float32)

        original_data = np.concatenate([C0_slice, DE_slice, T2_slice], axis=2)
        img_C0, img_DE, img_T2 = torch.chunk(image_transform(original_data), chunks=3, dim=0)

        test_C0 = torch.FloatTensor(1, 1, args_myops.dim, args_myops.dim).cuda()
        test_DE = torch.FloatTensor(1, 1, args_myops.dim, args_myops.dim).cuda()
        test_T2 = torch.FloatTensor(1, 1, args_myops.dim, args_myops.dim).cuda()

        seg = torch.FloatTensor(dim_z, args_myops.dim, args_myops.dim)

        for j in range(dim_z):
            img_C0_slice = slice_normalize(img_C0[j:j + 1, ...])
            img_DE_slice = slice_normalize(img_DE[j:j + 1, ...])
            img_T2_slice = slice_normalize(img_T2[j:j + 1, ...])

            test_C0.copy_(img_C0_slice.unsqueeze(0))
            test_DE.copy_(img_DE_slice.unsqueeze(0))
            test_T2.copy_(img_T2_slice.unsqueeze(0))

            # res_seg = model(test_C0, test_DE, test_T2)
            input = torch.cat([test_C0, test_DE, test_T2], 1)
            res_seg = Head1(backbone1(input))

            res_seg = res_seg["seg"]

            seg[j:j + 1, :, :].copy_(torch.argmax(res_seg, dim=1))

        seg = result_transform(seg)

        img_ct = (dim_x // 2, dim_y // 2)
        half_seg = args_myops.dim // 2

        res[:, img_ct[0] - half_seg: img_ct[0] + half_seg, img_ct[1] - half_seg: img_ct[1] + half_seg].copy_(seg)
        res = res.numpy().transpose(1, 2, 0)

        seg_map = nib.Nifti1Image(res, img_affine)

        nib.save(seg_map, 'test/test_' + str(epoch) + '/' + test_img.iloc[i]["file_name"] + '_result.nii.gz')

        print(test_img.iloc[i]["file_name"] + "_Successfully saved!")


def predict_multiple(args, args_myops):
    if not os.path.exists(args_myops.test_path):
        os.makedirs(args_myops.test_path)
    load_files = os.listdir(args_myops.load_path)
    load_backbones = sorted([i for i in load_files if "backbone" in i], reverse=True)
    load_heads = sorted([i for i in load_files if "head_myops" in i], reverse=True)
    model_list = list(zip(load_backbones, load_heads))
    for models in model_list:
        backbone = models[0]
        head = models[1]
        backbone_path = os.path.join(args_myops.load_path, backbone)
        head_path = os.path.join(args_myops.load_path, head)
        file_info = backbone.replace("backbone.pth", "")
        dice = float(file_info.split('_')[0])
        epoch = int(file_info.split('_')[1])
        print('--- start predicting ' + str(epoch) + ' ---')
        predict(args, args_myops, backbone_path, head_path, epoch)
        print('--- ' + str(epoch) + ' test done ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSCMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args_myops = config()
    predict_multiple(args, args_myops)
