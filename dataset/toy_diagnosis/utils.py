import os
import shutil
from tqdm import tqdm

train = 0
test = 0
root_dir = '/220019054/Dataset/US_finetune/toy_finetune/'
for i in os.listdir(root_dir+'train'):
    for j in os.listdir(root_dir+'train/'+i):
        train+=1
for i in os.listdir(root_dir+'test'):
    for j in os.listdir(root_dir+'test/'+i):
        test+=1 

print(train, test)
exit()
root_dir = '/220019054/Dataset/US_finetune/'
dst_dir=  '/220019054/Dataset/US_finetune/toy_finetune/'

dataset_list = ['BUSI', 'CEUS', 'ERUS', 'EUS', 'GBCU', 'kidneyUS', 'POCUS', 'LEPset']

for dataset in tqdm(dataset_list):
    if dataset == 'BUSI':
        shutil.copytree(root_dir+dataset+'/train/malignant', dst_dir+'train/breast_cancer')
        # shutil.copytree(root_dir+dataset+'/test/malignant', dst_dir+'test/breast_cancer')
    elif dataset == 'CEUS':
        shutil.copytree(root_dir+dataset+'/train/HCC', dst_dir+'train/liver_Hepatocellular_Carcinoma')
        # shutil.copytree(root_dir+dataset+'/test/HCC', dst_dir+'test/liver_Hepatocellular_Carcinoma')
    elif dataset == 'ERUS':
        shutil.copytree(root_dir+dataset+'/train/4', dst_dir+'train/colorectal_t4_cancer')
        # shutil.copytree(root_dir+dataset+'/test/4', dst_dir+'test/colorectal_t4_cancer')
    elif dataset == 'EUS':
        shutil.copytree(root_dir+dataset+'/train/HVD', dst_dir+'train/cardiac_valve_disease')
        # shutil.copytree(root_dir+dataset+'/test/HVD', dst_dir+'test/cardiac_valve_disease')
    elif dataset == 'GBCU':
        shutil.copytree(root_dir+dataset+'/train/malignant', dst_dir+'train/gallbladder_cancer')
        # shutil.copytree(root_dir+dataset+'/test/malignant', dst_dir+'test/gallbladder_cancer')
    elif dataset == 'kidneyUS':
        shutil.copytree(root_dir+dataset+'/train/Maligant_Tumor', dst_dir+'train/kidney_cancer')
        # shutil.copytree(root_dir+dataset+'/test/Maligant_Tumor', dst_dir+'test/kidney_cancer')
    elif dataset == 'POCUS':
        shutil.copytree(root_dir+dataset+'/train/Covid', dst_dir+'train/lung_Covid_19')
        # shutil.copytree(root_dir+dataset+'/test/Covid', dst_dir+'test/lung_Covid_19')
    if dataset == 'LEPset':
        shutil.copytree(root_dir+dataset+'/train/PC', dst_dir+'train/pancreas_cancer')
        # shutil.copytree(root_dir+dataset+'/test/PC', dst_dir+'test/pancreas_cancer')