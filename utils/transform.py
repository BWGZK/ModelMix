import torch
import random
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
from skimage.transform import resize
import PIL


class Max_Min_Normalization(object):
    def __call__(self, image):
        eps = 1e-8
        mn = image.min()
        mx = image.max()
        image_normalized = (image - mn) / (mx - mn + eps)       
        return image_normalized


class Zero_Mean_Unit_Variance_Normalization(object):
    def __call__(self, image):
        eps = 1e-8
        mean = image.mean()
        std = image.std()
        image_normalized = (image - mean) / (std + eps)   
        return image_normalized


class Truncate_and_Normalize(object):
    def __call__(self, image):
        # truncate
        Hist, _ = np.histogram(image, bins=int(image.max()))

        idexs = np.argwhere(Hist >= 20)
        idex_min = np.float32(0)
        idex_max = np.float32(idexs[-1, 0])

        image[np.where(image <= idex_min)] = idex_min
        image[np.where(image >= idex_max)] = idex_max

        # normalize
        sig = image[0, 0, 0]
        image = np.where(image != sig, image - np.mean(image[image != sig]), 0 * image)
        image = np.where(image != sig, image / np.std(image[image != sig] + 1e-7), 0 * image)
        return image


class RandomSizeCrop(object):
    def __init__(self, dim):
        self.crop_size = dim

    def __call__(self, image):
        scaler = np.random.uniform(0.7, 1.3)
        scale_size = int(self.crop_size * scaler)
        h_off = random.randint(0, image.shape[1] - 0 - scale_size)
        w_off = random.randint(0, image.shape[2] - 0 - scale_size)
        image = image[:, h_off:h_off+scale_size, w_off:w_off+scale_size]
        return image

# class RandomRotate(object):
#     def __init__(self, degrees, resample=False, expand=False, center=None):
#         self.degrees = degrees
#         self.resample = resample
#         self.expand = expand
#         self.center = center
    
#     @staticmethod
#     def get_params(degrees):
#         angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
#         return angle
    
#     def __call__(self, img):
#         angle = self.get_params(self.degrees)
#         rotated_img = F.rotate(img, angle, PIL.Image.NEAREST, self.expand, self.center)
#         rotated_img = rotated_img
#         return rotated_img
def one_hot(gt):
    gt_flatten = gt.flatten().astype(np.uint8)
    gt_onehot = np.zeros((gt_flatten.size, 6))
    gt_onehot[np.arange(gt_flatten.size), gt_flatten] = 1
    return gt_onehot.reshape(gt.shape[0], gt.shape[1], 6)

class RandomRotate(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
    
    @staticmethod
    def get_params(degrees):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        
        return angle
    def __call__(self, img):
        angle1 = self.get_params(self.degrees)
        #angle2 = self.get_params(self.degrees)
        rotated_img = F.rotate(img, angle1, PIL.Image.NEAREST, self.expand, self.center)
        #rotated_img_aux = F.rotate(img, angle2, PIL.Image.NEAREST, self.expand, self.center)
        #rotated_img_aux = rotated_img_aux.numpy()
        rotated_img = rotated_img.numpy()
        
        #beta = np.random.beta(0.5, 0.5, [1,1,1])
        
        #rotated_img[0:3]= (1-beta)*rotated_img_aux[0:3]+beta*rotated_img[0:3]
        
        #rotated_mask_onehot = one_hot(rotated_img[3])
        #rotated_mask_aux_onehot = one_hot(rotated_img[3])
        #mix_label = (1-beta)*rotated_mask_aux_onehot+beta*rotated_mask_onehot
        #rotated_img[3] = mix_label.argmax(-1).astype(np.uint8)

        x_min_cut = random.randint(0, img.shape[1])
        x_max_cut = min(x_min_cut+32, img.shape[1])
        y_min_cut = random.randint(0, img.shape[1])
        y_max_cut = min(y_min_cut+32, img.shape[1])
        rotated_img[:,x_min_cut:x_max_cut, y_min_cut:y_max_cut] = 0
        rotated_img = torch.from_numpy(rotated_img).float()
        
        return rotated_img

class Resize(object):
    def __init__(self, dim):
        super().__init__()

        self.crop_size = dim
    
    def __call__(self, image):
        
        image = image.numpy()
        
        C0_slice = image[:1,...]
        DE_slice = image[1:2,...]
        T2_slice = image[2:3,...]
        label_slice = image[3:,...]

        output_shape = (1, self.crop_size, self.crop_size)

        C0_resized = resize(C0_slice, output_shape, order=1, mode='constant', preserve_range=True)
        DE_resized = resize(DE_slice, output_shape, order=1, mode='constant', preserve_range=True)
        T2_resized = resize(T2_slice, output_shape, order=1, mode='constant', preserve_range=True)
        label_resized = resize(label_slice, output_shape, order=0, mode='edge', preserve_range=True)

        image = np.concatenate([C0_resized, DE_resized, T2_resized, label_resized], axis=0)
        image = torch.from_numpy(image).float()
        
        return image


class ToTensor(object):
    def __call__(self, image):
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = torch.from_numpy(image).float()
        return image


# image transform
class ImageTransform(object):
    def __init__(self, dim, stage):  
        self.dim = dim
        self.stage = stage

    def __call__(self, image):

        if self.stage == 'Train':
            transform = transforms.Compose([
                ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRotate((0,360)),
                # transforms.RandomRotation(90),
                RandomSizeCrop(self.dim),
                Resize(self.dim),
            ])     
        
        if self.stage == 'Valid':
            transform = transforms.Compose([
                ToTensor(),
                transforms.CenterCrop(self.dim),
            ])

        if self.stage == 'Test':
            transform = transforms.Compose([
                ToTensor(),
                transforms.CenterCrop(self.dim),
            ])

        return transform(image)


# label transform
class LabelTransform(object):
    def __init__(self, stage):
        
        self.stage = stage

    def __call__(self, label):
        
        # label = label.numpy()
        transformed_label = label

        # transformed_label = self.transform(label)

        if self.stage == 'Train':
            transformed_label = self.convert_onehot(transformed_label, 5) 
        else:
            pass
        
        return transformed_label

    def convert_onehot(self, label, num_class):
        label = label.long()
        if 5 in np.unique(label):
            num_class = 6
        label_onehot = torch.zeros((num_class, label.shape[1], label.shape[2]))
        label_onehot.scatter_(0, label, 1).float()
        return label_onehot[0:5,:,:]
            
    # def convert_onehot(self, label, num_class):
    #     label = label.long()
    #     label_onehot = torch.zeros((num_class, label.shape[1], label.shape[2]))
    #     label_onehot.scatter_(0, label, 1).float()
    #     return label_onehot 

    # 0 bg, 1 myo, 2 lv, 3 edema, 4 scar
    # def transform(self, gd):
    #     gd = np.where(gd == 200, 1, gd)
    #     gd = np.where(gd == 500, 2, gd)
    #     gd = np.where(gd == 600, 0, gd)
    #     gd = np.where(gd == 1220, 3, gd)
    #     gd = np.where(gd == 2221, 4, gd)
    #     gd = torch.from_numpy(gd).float()
    #     return gd
    def transform(self, gd):
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



# result transform
class ResultTransform(object):
    def __call__(self, seg):
        seg = np.round(seg)
        seg = seg.numpy()

        seg = np.where(seg == 1, 0, seg) # 0 bg myo, 2 lv, 3 edema, 4 scar
        seg = np.where(seg == 2, 0, seg) # 0 bg myo lv, 3 edema, 4 scar

        seg = np.where(seg == 3, 1220, seg)
        seg = np.where(seg == 4, 2221, seg)
        
        seg = torch.from_numpy(seg)
        
        return seg
