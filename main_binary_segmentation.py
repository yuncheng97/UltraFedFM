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

class TrainData(Dataset):
    def __init__(self, args, mode):
        self.args      = args
        self.mode      = mode
        self.samples   = [name for name in os.listdir(args.datapath+mode+'/image') if name[0]!="."]
        label_fraction = 1
        self.samples =  random.sample(self.samples, int(len(self.samples)*label_fraction))
        self.transform = A.Compose([
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(args.img_size, args.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(self.args.datapath+self.mode+'/image/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.args.datapath+self.mode+'/mask/'+name, cv2.IMREAD_GRAYSCALE)/255.0
        pair  = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)

class ValData(Dataset):
    def __init__(self, args, mode):
        self.args      = args
        self.mode      = mode
        self.samples   = [name for name in os.listdir(args.datapath+mode+'/image') if name[0]!="."]
        label_fraction = 1
        self.samples =  random.sample(self.samples, int(len(self.samples)*label_fraction))
        self.img_transform = A.Compose([
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(args.img_size, args.img_size),
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
        mask  = cv2.imread(self.args.datapath+self.mode+'/mask/'+name, cv2.IMREAD_GRAYSCALE)/255.0
        image = self.img_transform(image=image)['image']
        mask = self.mask_transform(image=mask)['image']
        return image, mask, name

    def __len__(self):
        return len(self.samples)

def bce_dice(pred, mask):
    ce_loss   = F.binary_cross_entropy_with_logits(pred, mask)
    # pred      = torch.sigmoid(pred)
    inter     = (pred*mask).sum(dim=(1,2))
    union     = pred.sum(dim=(1,2))+mask.sum(dim=(1,2))
    dice_loss = 1-(2*inter/(union+1)).mean()
    return ce_loss, dice_loss

class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # if not np.any(x):
        #     x[0][0] = 1.0
        # elif not np.any(y):
        #     y[0][0] = 1.0

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
            ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()
        if torch.sum(pred) == 0:
            pred[0][0][0][0] = 1
            # print(pred)
            # print(torch.sum(pred))
        # print(pred.shape)
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
            ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
            ).float()


        return torch.max(right_hd, left_hd)

def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.Tensor([1]).cuda()
        
    # IoU
    IoU = TP / (TP + FP + FN)
    # DICE
    DICE = 2 * IoU / (IoU + 1)

    pred  = pred.data.cpu().numpy().squeeze()
    gt    = gt.data.cpu().numpy().squeeze()
    gt    /= (gt.max() + 1e-8)
    pred  = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mae   = np.sum(np.abs(pred-gt))*1.0/(gt.shape[0]*gt.shape[1])



    
    return IoU.cpu().numpy(), DICE.cpu().numpy(), mae


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
        ENCODER = 'mae'
        ENCODER_WEIGHTS = args.pretrained
        ACTIVATION = 'sigmoid'
        self.model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, 
                                 in_channels=3, classes=1, activation=ACTIVATION)
        print('load pretrained weight from {}'.format(args.pretrained))
        logging.info('load pretrained weight from {}'.format(args.pretrained))

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                self.model.load_state_dict(checkpoint)
        # self.model.train(True)
        self.model.cuda()
        ## parameter
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epoch, eta_min=1e-6)
        warmup_epochs  = args.epoch // 10
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=self.scheduler)
        self.scheduler.step()
        # self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level='O2')
        if not args.eval and not args.plot:
            self.logger    = SummaryWriter(args.exp_path)
        self.best_mae   = 1
        self.best_dice  = 0
        self.best_acc   = 0
        self.best_epoch = 0
        self.best_f1    = 0

    def train(self):
        global_step = 0
        EARLY_STOPS = 100
        for epoch in range(100000):
            local_step = 0
            self.model.train()
            # if epoch+1 in [64, 96]:
            #     self.optimizer.param_groups[0]['lr'] *= 0.5
            #     self.optimizer.param_groups[1]['lr'] *= 0.5

            for image, mask in self.train_loader:
                image, mask = image.cuda().float(), mask.cuda().float()

                pred = self.model(image)
                pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=True)[:,0,:,:]
                # pred = pred.sigmoid()
                loss_ce, loss_dice = bce_dice(pred, mask)

                self.optimizer.zero_grad()
                # with apex.amp.scale_loss(loss_ce+loss_dice, self.optimizer) as scale_loss:
                loss = loss_ce + loss_dice
                loss.backward()
                self.optimizer.step()

                ## log
                global_step += 1
                local_step  += 1
                self.logger.add_scalar('lr'  , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.logger.add_scalars('loss', {'ce':loss_ce.item(), 'dice':loss_dice.item()}, global_step=global_step)
                if global_step % 10 == 0:
                    print(f'{datetime.now()} | epoch: {epoch+1:d}/{self.args.epoch:d} | step:{local_step:d}/{int(len(self.train_loader)):d} | lr={self.optimizer.param_groups[0]["lr"]:.6f} | ce={loss_ce.item():.6f} | dice={loss_dice.item():.6f}')
                    logging.info(f'{datetime.now()} | epoch: {epoch+1:d}/{self.args.epoch:d} | step:{local_step:d}/{int(len(self.train_data)):d} | lr={self.optimizer.param_groups[0]["lr"]:.6f} | ce={loss_ce.item():.6f} | dice={loss_dice.item():.6f}')
                    # print('%s | step:%d/%d/%d | lr=%.6f | ce=%.6f | dice=%.6f'%(datetime.now(), global_step, epoch+1, self.args.epoch, self.optimizer.param_groups[0]['lr'], loss_ce.item(), loss_dice.item()))
                    # logging.info('%s | step:%d/%d/%d | lr=%.6f | ce=%.6f | dice=%.6f'%(datetime.now(), global_step, epoch+1, self.args.epoch, self.optimizer.param_groups[0]['lr'], loss_ce.item(), loss_dice.item()))
            self.scheduler.step()

            self.val(self.val_loader, self.model, epoch, self.args.exp_path)

            if epoch - self.best_epoch > EARLY_STOPS:
                print (str(EARLY_STOPS), "epoches didn't improve, early stop.")
                print ("Best dice:", self.best_dice)
                break
            # if (epoch+1)%8==0:
            #     torch.save(self.model.state_dict(), self.args.savepath+'/model-'+str(epoch+1))

    def val(self, val_loader, model, epoch, save_path):
        # best_mae, best_dice, best_acc, best_epoch
        model.eval()
        with torch.no_grad():
            mae_sum  = 0
            iou_sum  = 0
            dice_sum = 0
            sen_sum  = 0
            spe_sum = 0
            acc_sum = 0
            seconds = 0
            dice_lst = []
            iou_lst = []
            mae_lst = []
            hd_lst = []
            hd_metric = HausdorffDistance()

            for image, mask, _ in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                image    = image.cuda()
                mask     = mask.cuda()

                start     = time.time()
                pred      = model(image)
                end       = time.time()
                seconds += end - start
  
                iou, dice, mae  = evaluate(pred, mask)
                hd = hd_metric.compute(pred, mask)
                hd = hd.numpy()
                dice_lst.append(dice)
                iou_lst.append(iou)
                mae_lst.append(mae)
                hd_lst.append(hd)
                mask_pred_show = (pred.squeeze().cpu().numpy())*255

            fps     = len(val_loader) / seconds

            dice = np.average(dice_lst)
            iou = np.average(iou_lst)
            mae = np.average(mae_lst)
            hd = np.average(hd_lst)
            if type(dice) is np.ndarray:
                dice = dice[0]
            if type(iou) is np.ndarray:
                iou = iou[0]
            if type(mae) is np.ndarray:
                mae = mae[0]
            if type(hd) is np.ndarray:
                hd = hd[0]
            self.logger.add_scalar('MAE', mae, global_step=epoch)
            self.logger.add_scalar('I0U', iou, global_step=epoch)
            self.logger.add_scalar('Dice', dice, global_step=epoch)
            # self.logger.add_scalar('HD', hd, global_step=epoch)
            # if mae < self.best_mae:
            #     self.best_mae   = mae
            #     self.best_epoch = epoch
            #     # torch.save(model.state_dict(), save_path+'/epoch_bestMAE.pth')
            #     print(f'best MAE {self.best_mae:.3f} epoch:{epoch}')
            #     logging.info(f'best MAE {self.best_mae:.3f} epoch:{epoch}')
                
            if dice > self.best_dice:
                self.best_dice   = dice
                self.best_epoch = epoch
                torch.save(model.state_dict(), save_path+'/epoch_bestDice.pth')
                print(f'best Dice {self.best_dice:.3f} (IOU: {iou:.3f}) epoch:{epoch}')
                logging.info(f'best Dice {self.best_dice:.3f} (IOU: {iou:.3f}) epoch:{epoch}')
                    
            print(f'#TEST#:  MAE: {mae:.3f}  IoU: {iou:.3f} Dice: {dice:.3f}  fps: {fps:.3f} ####   bestDice: {self.best_dice:.3f}')
            logging.info(f'#TEST#: MAE: {mae:.3f}  IoU: {iou:.3f} Dice: {dice:.3f} fps: {fps:.3f} ####  bestDice: {self.best_dice:.3f}')
    
    def eval(self, val_loader, model, save_path):
        model.eval()
        with torch.no_grad():
            Dice = 0
            IoU = 0
            Mae = 0
            HD = 0
            hd_metric = HausdorffDistance()
            with open(save_path+'/eval.txt', 'w') as f:
                for image, mask, name in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                    image    = image.cuda()
                    mask     = mask.cuda()
                    pred      = model(image)
                    iou, dice, mae  = evaluate(pred, mask)
                    hd = hd_metric.compute(pred, mask)
                    hd = hd.numpy()
                    IoU += iou.item()
                    Dice += dice.item()
                    Mae += mae.item()
                    HD += hd.item()
                
                IoU = IoU / len(val_loader)
                Dice = Dice / len(val_loader)
                Mae = Mae / len(val_loader)
                HD = HD / len(val_loader)
                print(f'MAE: {Mae} HD: {HD} IoU: {IoU} Dice: {Dice}')
 
    def eval_instance(self, val_loader, model, save_path):
        model.eval()
        with torch.no_grad():
            hd_metric = HausdorffDistance()
            with open(save_path+'/eval.txt', 'w') as f:
                for image, mask, name in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                    image    = image.cuda()
                    mask     = mask.cuda()
                    pred      = model(image)
                    iou, dice, mae  = evaluate(pred, mask)
                    hd = hd_metric.compute(pred, mask)
                    hd = hd.numpy()
                    line = f'Image name: {name}:  MAE: {mae} HD: {hd} IoU: {iou} Dice: {dice}' + '\n'
                    f.write(line)

    def plot(self, val_loader, model, save_path):
        model.eval()
        with torch.no_grad():
            for image, mask, name in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                name = name[0]
                image    = image.cuda()
                mask     = mask.cuda()
                pred      = model(image)
                pred[pred < 0.5]=0
                pred[pred > 0.5]=1
                pred       = pred.squeeze().cpu().numpy()*255
                if not os.path.exists(os.path.join(save_path,'figures')):
                    os.makedirs(os.path.join(save_path,'figures'), exist_ok=True)
                cv2.imwrite(os.path.join(save_path,'figures/', name), np.uint8(pred))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    , type=str,     default='../data/train'         )
    parser.add_argument('--savepath'    , type=str,     default='./out'                 )
    parser.add_argument('--model_name'  , type=str,     default='vit_base_patch16'      )
    parser.add_argument('--mode'        , type=str,     default='train'                 )
    parser.add_argument('--lr'          , type=float,   default=0.01                    )
    parser.add_argument('--img_size'    , type=int,     default=224                     )
    parser.add_argument('--epoch'       , type=int,     default=128                     )
    parser.add_argument('--batch_size'  , type=int,     default=2                       )
    parser.add_argument('--weight_decay', type=float,   default=5e-4                    )
    parser.add_argument('--momentum'    , type=float,   default=0.9                     )
    parser.add_argument('--nesterov'    , default=True                                  )
    parser.add_argument('--num_workers' , type=int,     default=4                       )
    parser.add_argument('--gpu_id'      , type=str,     default='1'                     )
    parser.add_argument('--pretrained'  , type=str,     default=None                    )
    parser.add_argument('--note'        , type=str,     default=None                    )
    parser.add_argument('--eval'        , action='store_true'                           )
    parser.add_argument('--eval_instance' , action='store_true'                         )
    parser.add_argument('--plot'        , action='store_true'                           )
    parser.add_argument('--resume'      , type=str,     default=None                    )
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.plot or args.eval:
        save_path          = os.path.join(args.savepath, args.note)
        args.exp_path = '/'.join(args.resume.split('/')[:-1])
    else:
        save_path          = os.path.join(args.savepath, args.note)
        current_timestamp  = datetime.now().timestamp()
        current_datetime   = datetime.fromtimestamp(current_timestamp+29220)  # different time zone
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        args.exp_path      = os.path.join(save_path, 'log_'+formatted_datetime)

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(args.exp_path, exist_ok=True)

    logging.basicConfig(filename=args.exp_path+'/log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    tables  = [[args.pretrained, save_path, args.lr, args.epoch, args.batch_size, args.weight_decay, args.note]]
    headers = ['pretrained''savepath', 'lr', 'epoch', 'batch_size', 'weight_decay', 'note']
    print('===training configures===')
    print(tabulate(tables, headers, tablefmt="grid", numalign="center"))
    logging.info('\n'+tabulate(tables, headers, tablefmt="github", numalign="center"))

    t    = Train(TrainData, ValData, args)
    if args.plot:
        print("Start svaing prediction results")
        t.plot(t.val_loader, t.model, args.exp_path)
    elif args.eval_instance:
        print("Start instance evaluating")
        t.eval_instance(t.val_loader, t.model, args.exp_path)
    elif args.eval:
        print("Start evaluating")
        t.eval(t.val_loader, t.model, args.exp_path)
    else:
        print("Start training")
        t.train()


