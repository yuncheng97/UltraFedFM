import os
import sys
import cv2
import time
import torch
import ctypes
import random
import logging 
import argparse
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt

libgcc_s = ctypes.CDLL('libgcc_s.so.1')
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

from tqdm import tqdm
from timm import create_model
from tabulate import tabulate
from datetime import datetime, timedelta
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from scipy.ndimage.morphology import distance_transform_edt as edt

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, SoftBCEWithLogitsLoss
from util.aop_estimation import angle_of_progression_estimation


class DSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        """
        :param y_pred: (BS,3,512,512)
        :param y_truth: (BS,512,512)
        :return:
        """
        y_pred_f = F.one_hot(y_pred.argmax(dim=1).long(), 3)
        y_pred_f = torch.flatten(y_pred_f, start_dim=0, end_dim=2)

        y_truth_f = F.one_hot(y_truth.long(), 3)
        # y_truth_f = y_truth.transpose(1,3)
        y_truth_f = torch.flatten(y_truth_f, start_dim=0, end_dim=2)

        dice1 = (2. * ((y_pred_f[:, 1:2] * y_truth_f[:, 1:2]).sum()) + self.smooth) / (
                y_pred_f[:, 1:2].sum() + y_truth_f[:, 1:2].sum() + self.smooth)
        dice2 = (2. * ((y_pred_f[:, 2:] * y_truth_f[:, 2:]).sum()) + self.smooth) / (
                y_pred_f[:, 2:].sum() + y_truth_f[:, 2:].sum() + self.smooth)

        dice1.requires_grad_(False)
        dice2.requires_grad_(False)
        return (dice1 + dice2) / 2

class TrainData(Dataset):
    def __init__(self, args, mode):
        self.args      = args
        self.mode      = mode
        self.samples   = [name for name in os.listdir(args.datapath+mode+'/image') if name[0]!="."]
        label_fraction = 1
        self.samples =  random.sample(self.samples, int(len(self.samples)*label_fraction))
        self.transform = A.Compose([
            A.Resize(args.img_size, args.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,),
            ToTensorV2()
        ])

    
    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(self.args.datapath+self.mode+'/image/'+name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.args.datapath+self.mode+'/mask/'+name, cv2.IMREAD_GRAYSCALE)
        mask[np.where(mask == 7)] = 1
        mask[np.where(mask == 8)] = 2
        pair  = self.transform(image=image, mask=mask)
        image, mask = pair['image'], pair['mask'] 
        mask = mask.type(torch.IntTensor)
        return image, mask

    def __len__(self):
        return len(self.samples)


class ValData(Dataset):
    def __init__(self, args, mode):
        self.args      = args
        self.mode      = mode
        self.samples   = [name for name in os.listdir(args.datapath+mode+'/image') if name[0]!="."]
        self.img_transform = A.Compose([
            # A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(args.img_size, args.img_size),
            A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,),
            ToTensorV2()
        ])
        self.mask_transform = A.Compose([
            A.Resize(args.img_size, args.img_size),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name  = self.samples[idx]
    
        image = cv2.imread(self.args.datapath+self.mode+'/image/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.args.datapath+self.mode+'/mask/'+name, cv2.IMREAD_GRAYSCALE)
        mask[np.where(mask == 7)] = 1
        mask[np.where(mask == 8)] = 2

        image = self.img_transform(image=image)['image']
        mask = self.mask_transform(image=mask)['image'][0]
        mask = mask.type(torch.IntTensor)
 

        return image, mask, name

    def __len__(self):
        return len(self.samples)




class Train(object):
    def __init__(self, TrainData, ValData, args):
        ## dataset
        self.args      = args 
        self.train_data    = TrainData(args, mode='train')
        self.val_data      = ValData(args, mode='test')
        self.train_loader  = DataLoader(self.train_data, batch_size=int(args.batch_size), pin_memory=True, shuffle=True, num_workers=args.num_workers)
        self.val_loader    = DataLoader(self.val_data, batch_size=1, pin_memory=True, shuffle=True, num_workers=args.num_workers)
        print('train dataset: ', len(self.train_data), 'val dataset: ', len(self.val_data))
        ## model
        CLASSES = args.nb_classes
        ENCODER = 'mae'
        ENCODER_WEIGHTS = args.pretrained
        ACTIVATION = 'sigmoid'
        self.model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, 
                                 in_channels=3, classes=CLASSES, activation=ACTIVATION)
        print('load pretrained weight from {}'.format(args.pretrained))
        logging.info('load pretrained weight from {}'.format(args.pretrained))

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                self.model.load_state_dict(checkpoint)
        self.model.cuda()
        ## parameter
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epoch, eta_min=1e-6)
        warmup_epochs  = args.epoch // 10
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=self.scheduler)
        self.scheduler.step()
        self.criterion = DiceLoss(mode='multiclass')
        self.evlauter = DSC()
        if not args.plot and not args.eval:
            self.logger    = SummaryWriter(args.exp_path)
        self.best_dice  = 0

    def train(self):
        global_step = 0
        EARLY_STOPS = 100
        for epoch in range(1000000):
            local_step = 0
            self.model.train()

            for image, mask in self.train_loader:
            
                image, mask = image.cuda().float(), mask.cuda().float()

                pred = self.model(image)
                # breakpoint()
                self.optimizer.zero_grad()
                loss = self.criterion(pred, mask)
                # loss = DiceLoss(pred, mask)
                loss.backward()
                self.optimizer.step()

                ## log
                global_step += 1
                local_step  += 1
                self.logger.add_scalar('lr'  , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.logger.add_scalar('loss', loss.item(), global_step=global_step)
                if global_step % 10 == 0:
                    print(f'{datetime.now()} | epoch: {epoch+1:d}/{self.args.epoch:d} | step:{local_step:d}/{int(len(self.train_loader)):d} | lr={self.optimizer.param_groups[0]["lr"]:.6f} | loss={loss.item():.6f}')
                    logging.info(f'{datetime.now()} | epoch: {epoch+1:d}/{self.args.epoch:d} | step:{local_step:d}/{int(len(self.train_loader)):d} | lr={self.optimizer.param_groups[0]["lr"]:.6f} | loss={loss.item():.6f}')
            self.scheduler.step()

            self.val(self.val_loader, self.model, epoch, self.args.exp_path)

            if epoch - self.best_epoch > EARLY_STOPS:
                print (str(EARLY_STOPS), "epoches didn't improve, early stop.")
                print ("Best dice:", self.best_dice)
                break
            # if (epoch+1)%8==0:
            #     torch.save(self.model.state_dict(), self.args.savepath+'/model-'+str(epoch+1))
        return

    def val(self, val_loader, model, epoch, save_path):
        model.eval()
        with torch.no_grad():
            dice_list = []
            seconds = 0
            for image, mask, _ in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                image    = image.cuda()
                mask     = mask.cuda()
                start     = time.time()
                pred      = model(image)
                end       = time.time()
                seconds += end - start
                pred = (pred > 0.5).int()
                dice_score = self.evlauter(pred, mask)
                dice_list.append(dice_score.cpu().numpy())
            fps     = len(val_loader) / seconds

            dice = np.average(dice_list)
            if type(dice) is np.ndarray:
                dice = dice[0]
            self.logger.add_scalar('Dice', dice, global_step=epoch)
                
            if dice > self.best_dice:
                self.best_dice   = dice
                self.best_epoch = epoch
                torch.save(model.state_dict(), save_path+'/epoch_bestDice.pth')
                print(f'best Dice {self.best_dice:.3f}  epoch:{epoch}')
                logging.info(f'best Dice {self.best_dice:.3f} epoch:{epoch}')
            print(f'\033[33m#TEST#\033[0m: Dice: {dice:.3f}  fps: {fps:.3f} ####   bestDice: {self.best_dice:.3f}')   
            logging.info(f'\033[33m#TEST#\033[0m: Dice: {dice:.3f} fps: {fps:.3f} ####  bestDice: {self.best_dice:.3f}')                

        return

    def eval(self, val_loader, model, save_path):
        model.eval()
        with torch.no_grad():
            dice_list = []
            abe_list = []
            seconds = 0
            with open(save_path+'/eval.txt', 'w') as f:
                for image, mask, name in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                    image    = image.cuda()
                    mask     = mask.cuda()
                    pred      = model(image)
                    pred = (pred > 0.5).int()
                    dice_score = self.evlauter(pred, mask)
                    dice_list.append(dice_score.cpu().numpy())

                    # aop estimation
                    pred = pred.squeeze(0).permute(1,2,0).cpu().numpy() # b,c,h,w -> c,h,w -> h,w,c

                    mask = F.one_hot(mask.long(), 3).squeeze(0).cpu().numpy() # b,h,w -> b,h,w,c -> h,w,c 

                    pred = (pred[:,:,1] + pred[:,:,2]*2)
                    mask = (mask[:,:,1] + mask[:,:,2]*2)
                    if np.unique(pred).shape[0] != 3:
                        pred_aop = 0
                    else:
                        pred_aop = angle_of_progression_estimation(pred, return_img=False)
                    if np.unique(mask).shape[0] != 3:
                        gt_aop = 0
                    else:
                        gt_aop = angle_of_progression_estimation(mask, return_img=False)
                    abe_score = np.abs(gt_aop - pred_aop)
                    abe_list.append(abe_score)
                    line = f'Image name:{name}, Dice:{dice_score}, Absolute error:{abe_score}' + '\n'
                    f.write(line)

                dice = np.average(dice_list)
                abe = np.average(abe_list)
                if type(dice) is np.ndarray:
                    dice = dice[0]
                if type(abe) is np.ndarray:
                    abe = abe[0]
                print(f'#TEST#: Overall Dice: {dice:.3f}  Overall Absolute error: {abe:.3f}')   
                f.write(f'#TEST#: Overall Dice: {dice:.3f}  Overall Absolute error: {abe:.3f}')                
        
        return

    def plot(self, val_loader, model, save_path):
        model.eval()
        with torch.no_grad():
            for image, mask, name in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                image    = image.cuda()
                mask     = mask.cuda()
                pred      = model(image)
                pred = (pred > 0.5).int()
                pred = pred.squeeze(0).permute(1,2,0).cpu().numpy() # b,c,h,w -> h,w,c
                pred = (pred[:,:,1]*1 + pred[:,:,2]*2)

                mask = F.one_hot(mask.long(), num_classes=3) #b,h,w -> b,h,w,c
                mask = mask.squeeze(0).cpu().numpy() # b,h,w,c - > h,w,c

                mask = (mask[:,:,1]*1 + mask[:,:,2]*2)
                # aop estimation
                if np.unique(mask).shape[0] != 3 or np.unique(pred).shape[0] != 3:
                    continue
                gt_aop, gt_aop_img = angle_of_progression_estimation(mask, return_img=True)
                pred_aop, pred_aop_img = angle_of_progression_estimation(pred, return_img=True)

                image = image.squeeze(0).permute(1,2,0).cpu().numpy()
                plt.figure(figsize=(19.2, 14.4), dpi=300)

                plt.subplot(1, 5, 1)
                plt.imshow(image)
                plt.title('Image')
                plt.axis('off')

                plt.subplot(1, 5, 2)
                plt.imshow(mask)
                plt.title('Ground Truth')
                plt.axis('off')

                plt.subplot(1, 5, 3)
                plt.imshow(pred)
                plt.title("Predicted Mask")
                plt.axis('off')

                plt.subplot(1, 5, 4)
                plt.imshow(gt_aop_img)
                plt.title(f'{round(gt_aop, 2)}')
                plt.axis('off')

                plt.subplot(1, 5, 5)
                plt.imshow(pred_aop_img)
                plt.title(f'{round(pred_aop, 2)}')
                plt.axis('off')

                plt.tight_layout()
                plt.show()
                if not os.path.exists(save_path+'/figures_multi'):
                    os.makedirs(save_path+'/figures_multi', exist_ok=True)
                plt.savefig(save_path+'/figures_multi/'+name[0], transparent = True, bbox_inches='tight', pad_inches=0.0)
                plt.close()
        return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    , type=str,     default='../data/train'     )
    parser.add_argument('--savepath'    , type=str,     default='./out'             )
    parser.add_argument('--model_name'  , type=str,     default='vit_base_patch16'  )
    parser.add_argument('--mode'        , type=str,     default='train'             )
    parser.add_argument('--lr'          , type=float,   default=0.01                )
    parser.add_argument('--img_size'    , type=int,     default=224                 )
    parser.add_argument('--epoch'       , type=int,     default=128                 )
    parser.add_argument('--nb_classes'  , type=int,     default=1                   )
    parser.add_argument('--batch_size'  , type=int,     default=2                   )
    parser.add_argument('--weight_decay', type=float,   default=5e-4                )
    parser.add_argument('--momentum'    , type=float,   default=0.9                 )
    parser.add_argument('--nesterov'    ,               default=True                )
    parser.add_argument('--num_workers' , type=int,     default=4                   )
    parser.add_argument('--gpu_id'      , type=str,     default='1'                 )
    parser.add_argument('--pretrained'  , type=str,     default=None                )
    parser.add_argument('--note'        , type=str,     default=None                )
    parser.add_argument('--eval'        ,               action='store_true'         )
    parser.add_argument('--plot'        ,               action='store_true'         )
    parser.add_argument('--resume'      , type=str,     default=None                )
    args = parser.parse_args()
    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not args.eval and not args.plot:
        save_path          = os.path.join(args.savepath, args.note)
        current_timestamp  = datetime.now().timestamp()
        current_datetime   = datetime.fromtimestamp(current_timestamp+29220)  # different time zone
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        args.exp_path      = os.path.join(save_path, 'log_'+formatted_datetime)

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(args.exp_path, exist_ok=True)
    else:
        save_path          = os.path.join(args.savepath, args.note)
        args.exp_path = '/'.join(args.resume.split('/')[:-1])
    

    logging.basicConfig(filename=args.exp_path+'/log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')

    tables  = [[args.pretrained, save_path, args.lr, args.epoch, args.batch_size, args.weight_decay, args.note]]
    headers = ['pretrained''savepath', 'lr', 'epoch', 'batch_size', 'weight_decay', 'note']
    print('===training configures===')
    print(tabulate(tables, headers, tablefmt="grid", numalign="center"))
    logging.info('\n'+tabulate(tables, headers, tablefmt="github", numalign="center"))

    t    = Train(TrainData, ValData, args)
    if args.plot:
        t.plot(t.val_loader, t.model, args.exp_path)
    elif args.eval:
        t.eval(t.val_loader, t.model, args.exp_path)
    else:
        t.train()


