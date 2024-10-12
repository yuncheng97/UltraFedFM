import os
from tqdm import tqdm
import shutil

train = 0
test = 0
root_dir = '/220019054/Dataset/US_finetune/segmentation/toy_seg/'
for i in os.listdir(root_dir+'train'):
    for j in os.listdir(root_dir+'train/'+i):
        train+=1
for i in os.listdir(root_dir+'test'):
    for j in os.listdir(root_dir+'test/'+i):
        test+=1 

print(train, test)
exit()

for i in tqdm(os.listdir('/220019054/Dataset/US_finetune/segmentation/NUS/test/image')):
    shutil.copy('/220019054/Dataset/US_finetune/segmentation/NUS/test/image/'+i, '/220019054/Dataset/US_finetune/segmentation/toy_seg/test/image/nus_'+i)
for i in tqdm(os.listdir('/220019054/Dataset/US_finetune/segmentation/NUS/test/mask')):
    shutil.copy('/220019054/Dataset/US_finetune/segmentation/NUS/test/mask/'+i, '/220019054/Dataset/US_finetune/segmentation/toy_seg/test/mask/nus_'+i)
for i in tqdm(os.listdir('/220019054/Dataset/US_finetune/segmentation/NUS/train/image')):
    shutil.copy('/220019054/Dataset/US_finetune/segmentation/NUS/train/image/'+i, '/220019054/Dataset/US_finetune/segmentation/toy_seg/train/image/nus_'+i)
for i in tqdm(os.listdir('/220019054/Dataset/US_finetune/segmentation/NUS/train/mask')):
    shutil.copy('/220019054/Dataset/US_finetune/segmentation/NUS/train/mask/'+i, '/220019054/Dataset/US_finetune/segmentation/toy_seg/train/mask/nus_'+i)