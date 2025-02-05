import cv2
import numpy as np
import os
import csv
import torchvision.transforms.functional as f
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='script to compute scores')
parser.add_argument('-d0','--dir_origin', default='/220019054/Dataset/US_Pretrain/25_clients/split_1', help='Path to original images', type=str)
parser.add_argument('-d1','--dir_structure', default='./dataset/coco/structure/test', help='Path to structure images', type=str)
parser.add_argument('-d2','--dir_score', default='/220019054/Dataset/US_Pretrain/25_clients/split_1_label', help='Path to output score', type=str)

opt = parser.parse_args()

def laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img

def cal_texture(img, crop_sz=16, step=16):
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
            score = laplacian(crop_img).mean()
            score = round(score, 2)
            patch_scores.append(score)
    return patch_scores

def cal_structure(img, crop_sz=16, step=16):
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
            crop_img = img[x:x + crop_sz, y:y + crop_sz,:]
            score = round(crop_img.mean()+0.1, 2)
            patch_scores.append(score)
    return patch_scores

if __name__ == '__main__':
    for img_file in os.listdir(opt.dir_origin):
        print(img_file)
        with open(os.path.join(opt.dir_score, img_file), 'w') as f1:
            with open(os.path.join(opt.dir_origin, img_file)) as f2:
                lines = f2.readlines()
                for line in tqdm(lines):
                    img_name = line.strip('\n')
                    img = cv2.imread(img_name)
                    img = cv2.resize(img, (224, 224))
                    score = cal_texture(img, 16, 16)
                    f1.write(str(score)+'\n')
    print("finish")