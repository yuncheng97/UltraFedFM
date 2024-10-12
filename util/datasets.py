# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import cv2
import PIL
import torch
import random
import argparse
import numpy as np
import torch.utils.data as data


from PIL import Image, ImageFilter
from timm.data import create_transform
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

RETINA_MEAN = (0.5007, 0.5010, 0.5019)
RETINA_STD = (0.0342, 0.0535, 0.0484)

class PartialImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, subset_fraction=0.2):
        super(PartialImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        
        # Calculate the subset size
        self.subset_size = int(len(self.samples) * subset_fraction)
        
        # Generate indices for the subset
        self.indices = torch.randperm(len(self.samples))[:self.subset_size]
        
    def __len__(self):
        return self.subset_size
    
    def __getitem__(self, index):
        actual_index = self.indices[index]
        path, target = self.samples[actual_index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def build_partial_dataset(is_train, subset_fraction, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'test')
    dataset = PartialImageFolder(root, transform=transform, target_transform=None, subset_fraction=subset_fraction)


    print(dataset)

    return dataset

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'test')
    dataset = datasets.ImageFolder(root, transform=transform)
    # folders_to_include = ['BUSI', 'ERUS']

    # 初始化自定义数据集
    # dataset = SelectiveImageFolder(root, folders_to_include=folders_to_include, transform=transform) 

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # mean = [0.162, 0.163, 0.165]
    # std = [0.180, 0.179, 0.182] 

    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

if __name__ == '__main__':

    # simple augmentation
    # transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomApply(
    #             [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5
    #         ),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # folders_to_include = ['BUV', 'CUV']
    # dataset_train = SelectiveImageFolder('/220019054/Dataset/US_Pretrain/train', folders_to_include=folders_to_include, transform=transform_train)
    # print(dataset_train)
    parser = argparse.ArgumentParser('Fed-MAE pre-training', add_help=False)
    # parser.add_argument('--data_path', type=str,
    #                     help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # parser.add_argument('--split_type', type=str,
    #                     help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # parser.add_argument('--single-client', type=str,
    #                     help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    args = parser.parse_args()
    args.data_path = '/220019054/Dataset/US_Pretrain'
    args.input_size = 224

    # with open('/220019054/Dataset/US_Pretrain/25_clients/split_1_label/client_0.txt') as f:
    #     lines = f.readlines()
    #     List = list({line.strip('\n').strip('[').strip(']') for line in lines})
    #     breakpoint()
    #     # label = List[0]
    #     # label = list({float(i) for i in label.split(',')})
    #     breakpoint()
    dataset = USDataset(args)
    sample, de_sample, label = dataset[0]
    breakpoint()