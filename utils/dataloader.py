import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import random
import nibabel as nib
from torch.utils.data import Dataset
from utils.transform import Truncate_and_Normalize, ImageTransform, LabelTransform, \
    Zero_Mean_Unit_Variance_Normalization


class CrossModalDataLoader(Dataset):

    def __init__(self, path, file_name, dim, max_iters=None, stage='Train'):

        self.path = path
        self.crop_size = dim
        self.stage = stage
        self.Img = sorted([item.strip().split() for item in open(self.path + file_name)])
        if len(self.Img) == 85:
            for i in range(18):
                self.Img[i] = (self.Img[i], False)
            for i in range(18, len(self.Img), 1):
                self.Img[i] = (self.Img[i], True)
            if max_iters != None:
                unlabeled_img = []
                for j in range(20):
                    k = random.randint(18,85)
                    unlabeled_img.append(self.Img[k])
                self.Img = self.Img[0:18] * int(np.ceil(float(max_iters) / len(self.Img)))+unlabeled_img
            train_set = True
        else:
            for i in range(len(self.Img)):
                self.Img[i] = (self.Img[i], False)
            train_set = False

        self.files = []

        for index in range(len(self.Img)):
            item = self.Img[index][0]
            unlabel_tag = self.Img[index][1]
            img_path, gt_path, img_index = item

            C0_path = img_path + '_C0.nii.gz'
            DE_path = img_path + '_DE.nii.gz'
            T2_path = img_path + '_T2.nii.gz'
            label_path = gt_path + '_gd.nii.gz'

            C0_file = os.path.join(self.path, C0_path)
            DE_file = os.path.join(self.path, DE_path)
            T2_file = os.path.join(self.path, T2_path)
            label_file = os.path.join(self.path, label_path)
            if train_set and unlabel_tag:
                label_file = label_file + "unlabel_tag"

            self.files.append({
                "C0": C0_file,
                "DE": DE_file,
                "T2": T2_file,
                "label": label_file,
                "index": int(img_index)
            })

        self.image_transform = ImageTransform(self.crop_size, self.stage)
        self.label_transform = LabelTransform(self.stage)
        self.truncate = Truncate_and_Normalize()
        self.normalize = Zero_Mean_Unit_Variance_Normalization()

    def transform_label_values(self, gd):
        gd = np.round(gd)
        if 3000 not in np.unique(gd):
            gd = np.where(gd == 200, 1, gd)
            gd = np.where(gd == 500, 2, gd)
            gd = np.where(gd == 600, 0, gd)
            gd = np.where(gd == 1220, 3, gd)
            gd = np.where(gd == 2221, 4, gd)
        else:
            gd = np.where(gd == 0, 5, gd)
            gd = np.where(gd == 200, 1, gd)
            gd = np.where(gd == 500, 2, gd)
            gd = np.where(gd == 600, 0, gd)
            gd = np.where(gd == 1220, 3, gd)
            gd = np.where(gd == 2221, 4, gd)
            gd = np.where(gd == 3000, 0, gd)
            if len(np.unique(gd)) == 2:
                gd = np.where(gd == 5, 0, gd)
        # gd = torch.from_numpy(gd).float()
        return gd

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        file_path = self.files[index]

        # get raw data
        C0_raw = nib.load(file_path["C0"])
        DE_raw = nib.load(file_path["DE"])
        T2_raw = nib.load(file_path["T2"])
        if "unlabel_tag" in file_path["label"]:
            file_path["label"] = file_path["label"].replace("unlabel_tag", "")
            unlabel_tag = True
        else:
            unlabel_tag = False
        gd_raw = nib.load(file_path["label"])
        img_index = file_path["index"]

        # get data [x,y,z] & normalize
        C0_img = self.truncate(C0_raw.get_fdata())
        DE_img = self.truncate(DE_raw.get_fdata())
        T2_img = self.truncate(T2_raw.get_fdata())
        gd_img = gd_raw.get_fdata()

        # cut slice [x,y,1] -> [x,y,4]
        C0_slice = C0_img[:, :, img_index:img_index + 1].astype(np.float32)
        DE_slice = DE_img[:, :, img_index:img_index + 1].astype(np.float32)
        T2_slice = T2_img[:, :, img_index:img_index + 1].astype(np.float32)
        label_slice = gd_img[:, :, img_index:img_index + 1].astype(np.float32)
        label_slice = self.transform_label_values(label_slice)

        image = np.concatenate([C0_slice, DE_slice, T2_slice, label_slice], axis=2)

        # [4,H,W]
        image_transformed = self.image_transform(image)
        img_C0, img_DE, img_T2, label = torch.chunk(image_transformed, chunks=4, dim=0)

        img_C0 = self.normalize(img_C0)
        img_DE = self.normalize(img_DE)
        img_T2 = self.normalize(img_T2)

        # label transform [class,H,W]
        label = self.label_transform(label)

        indicator = torch.where(label.sum((-2, -1), keepdim=True) > 1, 1, 0)
        indicator = indicator.expand(-1, 192, 192)
        if unlabel_tag:
            label[:, :, :] = 0

        return img_C0, img_DE, img_T2, label, indicator
