import SimpleITK as sitk
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils import data
import nibabel as nib
import os
import data.transforms as T

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header

class seg_train(data.Dataset):
    def __init__(self, img_paths, lab_paths, transforms):
        self._transforms = transforms
        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}
        for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
            img_path_str = str(img_path)
            if "unlabel_tag" in img_path_str:
                img_path_str = img_path_str.replace("unlabel_tag","")
                img = self.read_image(img_path_str)
                img_name = img_path.split("/")[-1]
                self.img_dict.update({img_name : img})
                lab = self.read_label(str(lab_path))
                lab_name = lab_path.split("/")[-1]
                self.lab_dict.update({lab_name : lab})
            else:
                img = self.read_image(str(img_path))
                img_name = img_path.split("/")[-1]
                self.img_dict.update({img_name : img})
                lab = self.read_label(str(lab_path))
                lab_name = lab_path.split("/")[-1]
                self.lab_dict.update({lab_name : lab})
            assert img[0].shape[2] == lab[0].shape[2]
            self.examples += [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]
            
    def __getitem__(self, idx):
        img_name, lab_name, Z, X, Y = self.examples[idx]
        if Z != -1:
            img = self.img_dict[img_name][Z, :, :]
            lab = self.lab_dict[lab_name][Z, :, :]
        elif X != -1:
            img = self.img_dict[img_name][:, X, :]
            lab = self.lab_dict[lab_name][:, X, :]
        elif Y != -1:
            img = self.img_dict[img_name][0][:, :, Y]
            scale_vector_img = self.img_dict[img_name][1]
            lab = self.lab_dict[lab_name][0][:, :, Y]
            scale_vector_lab = self.lab_dict[lab_name][1]
        else:
            raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
        img = np.expand_dims(img, 0)
        lab = np.expand_dims(lab, 0)

        # repeat to 3 channels
        img = np.repeat(img, 3, 0)
        # lab = np.repeat(lab, 3, 0)
        lab_values = np.unique(lab)
        if "unlabel_tag" in img_name:
            lab[0] = np.zeros_like(lab[0])+4
        target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'lab_values':lab_values, 'orig_size': lab.shape}
        if self._transforms is not None:
            img, target = self._transforms([img, scale_vector_img], [target,scale_vector_lab])
        return img, target

    def read_image(self, img_path):
        img_dat = load_nii(img_path)
        img = img_dat[0]
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        img = img.astype(np.float32)
        return [(img-img.mean())/img.std(), scale_vector]

    def read_label(self, lab_path):
        lab_dat = load_nii(lab_path)
        lab = lab_dat[0]
        pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        # cla = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
        return [lab, scale_vector]

    def __len__(self):
        return len(self.examples)

def make_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize()
    ])

    if image_set == 'train':
        # return T.Compose([T.RandomHorizontalFlip(),T.RandomCrop([256, 256]), T.ToTensor()])
        return T.Compose([
            T.NoRescale(),
            T.RandomHorizontalFlip(),
            T.RandomRotate((0,360)),
            # T.CenterRandomCrop([0.7,1]),
            T.PadOrCropToSize([256,256]),
            # T.RandomColorJitter(),
            T.Resize([256,256]),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            T.NoRescale(),
            # T.Resize([256,256]),
            T.CenterPadOrCropToSize([256,256]),
            normalize])


    raise ValueError(f'unknown {image_set}')




def build(image_set, args, dataset_name):
    if dataset_name == "MSCMR":
        root = Path('/data/kzhang99/ModelMix/' + args.MSCMR_dataset)
    if dataset_name == "ACDC":
        root = Path('/data/kzhang99/ModelMix/' + args.MSCMR_dataset)
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels"),
        "val": (root / "val" / "images", root / "val" / "labels"),
    }
    if image_set == "train":
        img_folder, lab_folder = PATHS[image_set]
        img_paths = [os.path.join(img_folder, i) for i in sorted(os.listdir(img_folder))]
        lab_paths = [os.path.join(lab_folder, i) for i in sorted(os.listdir(lab_folder))]
        n = len(img_paths)
        label_num = 5
        if dataset_name == "MSCMR":
            for i in range(label_num,n,1):
                img_paths[i] = str(img_paths[i])+"unlabel_tag"
            img_paths = img_paths[0:label_num]*40+img_paths[label_num:n]
            lab_paths = lab_paths[0:label_num]*40+lab_paths[label_num:n]
            dataset_train = seg_train(img_paths, lab_paths, transforms=make_transforms(image_set))
            return dataset_train
        if dataset_name == "ACDC":
            tuples = list(zip(img_paths, lab_paths))
            random.shuffle(tuples)
            img_paths_shuffled = [i[0] for i in tuples]
            lab_paths_shuffled = [i[1] for i in tuples]
            for i in range(label_num,n,1):
                img_paths_shuffled[i] = str(img_paths_shuffled[i])+"unlabel_tag"
            # img_paths = img_paths[0:label_num]*40+img_paths[label_num:n]
            # lab_paths = lab_paths[0:label_num]*40+lab_paths[label_num:n]
            img_paths = img_paths_shuffled[0:label_num]*80+img_paths_shuffled[label_num:n]
            lab_paths = lab_paths_shuffled[0:label_num]*80+lab_paths_shuffled[label_num:n]
            dataset_train = seg_train(img_paths, lab_paths, transforms=make_transforms(image_set))
            return dataset_train
    elif image_set == "val":
        img_folder, lab_folder = PATHS[image_set]
        img_paths = [os.path.join(img_folder, i) for i in sorted(os.listdir(img_folder))]
        lab_paths = [os.path.join(lab_folder, i) for i in sorted(os.listdir(lab_folder))]
        dataset_val = seg_train(img_paths, lab_paths, transforms=make_transforms(image_set))
        return dataset_val
