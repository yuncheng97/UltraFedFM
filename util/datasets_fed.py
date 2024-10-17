# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import cv2
import torch
import random
import argparse
import numpy as np
import torch.utils.data as data


from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image, ImageFilter

RETINA_MEAN = (0.5007, 0.5010, 0.5019)
RETINA_STD = (0.0342, 0.0535, 0.0484)


def create_dataset_and_evalmetrix(args, mode='pretrain'):

    args.dis_cvs_files = os.listdir(os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type))
    
    args.clients_with_len = {}
    
    for single_client in args.dis_cvs_files:
        img_paths = list({line.strip().split(',')[0] for line in
                            open(os.path.join(args.data_path, f'{args.n_clients}_clients',
                                            args.split_type, single_client))})
        args.clients_with_len[single_client] = len(img_paths)

        
class CustomDataAugmentation(object):
    def __init__(self, args):
        # mean, std = (0.162, 0.163, 0.165), (0.180, 0.179, 0.182)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.crop = transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))

        self.args = args
        self.p = 0.5

    def __call__(self, img):        
        # Check if the horizontal flip will be applied
        # flip_applied = random.random() < self.p

        # Apply the horizontal flip to the image if necessary
        # if flip_applied:
        img = self.crop(img)
        img = self.flip(img)

        de_img = img.copy()
        # add low quality noise
        was_applied = random.random() < self.p
        if was_applied:
            transforms_list = []
            if random.choice([True, False]):
                transforms_list.append('motion_blur')
            if random.choice([True, False]):
                transforms_list.append('gaussian_blur')
            if random.choice([True, False]):
                transforms_list.append('saltpepper_noise')
            
            # 如果没有选择任何变换，强制选择一种
            if not transforms_list:
                transforms_list.append(random.choice(['motion_blur', 'saltpepper_noise', 'gaussian_blur']))

            # print(f'applied {transforms_list} transforms to image')
            for transform in transforms_list:
                if transform == 'motion_blur':
                    de_img = self.motion_blur(de_img)
                if transform == 'saltpepper_noise':
                    de_img = self.saltpepper_noise(de_img)
                if transform == 'gaussian_blur':
                    de_img = de_img.filter(ImageFilter.GaussianBlur(1.1))

        label = self.cal_texture(np.array(img), 16, 16)
        # breakpoint()
        label = torch.tensor(label)


        img = self.totensor(img)
        img = self.normalize(img)
        
        de_img = self.totensor(de_img)
        de_img = self.normalize(de_img)
            # label = self.flip_label(label)
            # label = self.totensor(label)
        
        return img, de_img, label

    def motion_blur(self, image, degree=24, angle=45):
        image = np.array(image)
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        blurred_image = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred_image, blurred_image, 0, 255, cv2.NORM_MINMAX)
        blurred_image = np.array(blurred_image, dtype=np.uint8)
        blurred_image = Image.fromarray(blurred_image.astype('uint8')).convert('RGB')  # numpy转图片
        return blurred_image

    def saltpepper_noise(self, image, density=0.1):
        image = np.array(image)  # 图片转numpy
        h, w, c = image.shape
        Nd = density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        image[mask == 0] = 0  # 椒
        image[mask == 1] = 255  # 盐
        image = Image.fromarray(image.astype('uint8')).convert('RGB')  # numpy转图片
        return image

    def laplacian(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        mask_img = cv2.convertScaleAbs(laplac)
        return mask_img

    def cal_texture(self, img, crop_sz=16, step=16):
        h, w, c = img.shape
        h_space = np.arange(0, h - crop_sz + 1, step)
        w_space = np.arange(0, w - crop_sz + 1, step)
        index = 0
        num_h = 0
        patch_scores = []
        for x in h_space:
            num_h += 1
            num_w = 0
            for y in w_space:
                num_w += 1
                index += 1
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                score = self.laplacian(crop_img).mean()
                score = round(score, 2)
                patch_scores.append(score)
        return patch_scores


class DatasetFLPretrain(data.Dataset):
    """ data loader for pre-training """
    def __init__(self, args):    
                

        cur_client_path       = os.path.join(args.data_path, f'{args.n_clients}_clients', 
                                        args.split_type, args.single_client)              # 25_clients/split_1/client_0
        # cur_client_label_path = os.path.join(args.data_path, f'{args.n_clients}_clients', 
        #                                 args.split_type+'_label', args.single_client)  # 25_clients/split_1_label/client_0

        self.img_paths        = list({line.strip().split(',')[0] for line in open(cur_client_path)})
        # self.labels           = list({line.strip('\n').strip('[').strip(']') for line in open(cur_client_label_path)})

        self.transform        = CustomDataAugmentation(args)
        self.args = args
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # breakpoint()
        index = index % len(self.img_paths)
        path =  os.path.join(self.args.data_path, self.img_paths[index])
        
        # label = self.labels[index]
        # label = list({float(i) for i in label.split(',')})
        # label_len = len(label)
        # breakpoint()
        # label = np.array(label).reshape(1, label_len)
        
        img = Image.open(path).convert("RGB")

        
        if self.transform is not None:
            sample, de_sample, target = self.transform(img)
        

            
        return sample, de_sample, target

    def __len__(self):
        return len(self.img_paths)