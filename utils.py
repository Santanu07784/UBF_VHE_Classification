# coding:utf-8

import numpy as np
import torch
import glob
from PIL import Image
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time # time measurement
import datetime # date representation
import sys # write function
import os
import shutil

def save_files(input_dataset,ver,src_file):
    
    dest_file = 'Result/%s_v%03d'%(input_dataset,ver)
    for src_file in src_file:
        if os.path.exists('%s/%s'%(dest_file,src_file)):
            os.remove('%s/%s'%(dest_file,src_file))
        shutil.copy(src_file, dest_file)
class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}

    def log(self, losses=None):
        self.mean_period = (time.time() - self.prev_time)
        self.prev_time = time.time()
        
        batches_done = (self.epoch-1) * self.batches_epoch + self.batch
        batches_left = self.n_epochs * self.batches_epoch - batches_done
        time_left = datetime.timedelta(seconds = batches_left*self.mean_period)
        
        sys.stdout.write('\rEpoch %03d/%03d Batch [%04d/%04d] ETA [%s]-- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch, time_left))
        
        #for i, loss_name in enumerate(losses.keys()):
        #    if loss_name not in self.losses:
        #        self.losses[loss_name] = np.zeros([self.n_epochs, self.batches_epoch])
        #        self.losses[loss_name][self.epoch-1,self.batch-1] = losses[loss_name].item()
        #   else:
        #        self.losses[loss_name][self.epoch-1,self.batch-1] = losses[loss_name].item()

        #    sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name][self.epoch-1,self.batch-1]))
        
        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            
            sys.stdout.write('\n')
            print('\rEpoch %03d/%03d Batch [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))
            for i, loss_name in enumerate(losses.keys()):
                print('%s: %.4f -- ' % (loss_name, np.mean(self.losses[loss_name], axis=1)[self.epoch-1]))
            
            self.epoch += 1
            self.batch = 1
            
        else:
            self.batch += 1
        
        return self.losses
 

# Utils
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


      
import torch.nn as nn
from torch.autograd import Variable  
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  
            input = input.transpose(1,2)    
            input = input.contiguous().view(-1,input.size(2))  
        target = target.view(-1,1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()     

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()
def calc_loss(pred, target, metrics, bce_weight = 0.5):

    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    #pred = (pred > 0.5).float()
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss    

metrics = defaultdict(float)

# Plot
def boxgraph(ax,label,pred):
    precision = eval_precision(label, pred)
    recall = eval_recall(label, pred)
    iou = eval_iou(label, pred)
    dice_sco = eval_dice(label, pred)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('%.3f'%height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize = 8)
    plt.ylim([0.,1.]) 
    score = [precision,recall,iou,dice_sco]
    autolabel(ax.bar(range(len(score)), score))
    ax.set_xticks([0,1,2,3])
    plt.xticks(rotation=15)
    ax.set_xticklabels(['precision','recall','iou','dice'])

#Evaluations
def eval_precision(y_real, y_pred):
    if np.sum(y_pred) == 0:
        out = 0.
    else:
        out = np.sum(y_pred[y_real == 1]) / np.sum(y_pred) * 1.0
    return out

def eval_precision_ave(y_real, y_pred, name):
    outl = 0

    for idx in range(y_real.shape[0]):
        y_r = torch.squeeze(y_real[idx,:,:]).detach().numpy()
        y_p = torch.squeeze(y_pred[idx,:,:]).detach().numpy()
        if np.sum(y_p) == 0:
            outl+=0
        else:
            outl+=(np.sum(y_p[y_r == 1]) / np.sum(y_p) * 1.0)
    out = outl 

    return out

def eval_recall(y_real, y_pred):
    if np.sum(y_real) == 0:
        out = 0.
    else:
        out = np.sum(y_real[y_pred == 1]) / np.sum(y_real) * 1.0
    return out

def eval_recall_ave(y_real, y_pred, name):
    outl = list()
    for idx in range(y_real.shape[0]):

        y_r = torch.squeeze(y_real[idx,:,:]).detach().numpy()
        y_p = torch.squeeze(y_pred[idx,:,:]).detach().numpy()
    
        if np.sum(y_r) == 0:
            outl.append(0)
        else:
            outl.append(np.sum(y_r[y_p == 1]) / np.sum(y_r) * 1.0)
    out = sum(outl) 
    return out

def eval_accuracy(y_real, y_pred):
    i_all = y_pred.shape[0] * y_pred.shape[1]
    i_inter = np.sum(y_pred[y_real == 1])
    i_a = np.sum(y_real)
    i_b = np.sum(y_pred)
    if i_all == 0:
        out = 0.
    else:
        out = 1 + 2*i_inter/i_all - (i_a+i_b)/i_all * 1.0
    return out

def eval_iou(y_real, y_pred):
    i_inter = np.sum(y_pred[y_real == 1])
    i_a = np.sum(y_real)
    i_b = np.sum(y_pred)
    if i_a + i_b - i_inter == 0:
        out = 0.
    else:
        out = i_inter / (i_a + i_b - i_inter) * 1.0
    return out

def eval_iou_ave(y_real, y_pred, name):
    outl = 0
    for idx in range(y_real.shape[0]):

        y_r = torch.squeeze(y_real[idx,:,:]).detach().numpy()
        y_p = torch.squeeze(y_pred[idx,:,:]).detach().numpy()
        i_inter = np.sum(y_p[y_r == 1])
        i_a = np.sum(y_r)
        i_b = np.sum(y_p)
        if i_a + i_b - i_inter == 0:
            outl += 0
        else:
            outl += (i_inter / (i_a + i_b - i_inter) * 1.0)
    out = outl 
    return out

def eval_dice(y_real, y_pred):
    i_inter = np.sum(y_pred[y_real == 1])
    i_a = np.sum(y_real)
    i_b = np.sum(y_pred)
    if i_a + i_b == 0:
        out = 0.
    else:
        out = 2 * i_inter / (i_a + i_b) * 1.0
    return out

def eval_dice_ave(y_real, y_pred,name):
    dice = 0
    for idx in range(y_real.shape[0]):

        y_r = torch.squeeze(y_real[idx,:,:]).detach().numpy()
        y_p = torch.squeeze(y_pred[idx,:,:]).detach().numpy()
        i_inter = np.sum(y_p[y_r == 1])
        i_a = np.sum(y_r)
        i_b = np.sum(y_p)
    
        if i_a + i_b == 0:
            dice += 0
        else:
            dice+= (2 * i_inter / (i_a + i_b))
    out = dice 

    return out
