# -*- coding: utf-8 -*-
from utils import *

import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
from torch.autograd import Variable
import timm
from tqdm import tqdm
import ttach as tta
import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from time import sleep
import argparse
import time

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--name", default='sample', type=str) 
parser.add_argument('--dataroot', default='datasets', type=str) 
parser.add_argument('--results_dir', default='results', type=str) 
parser.add_argument('--checkpoints_dir', default='checkpoints', type=str) 
parser.add_argument('--n_epochs', default=1000, type=int)
parser.add_argument("-d", "--data", required=True)  # e.g., 'HE','UBF','VHE','UBF_VHE'
parser.add_argument("-n", "--network", default='resnet')  # e.g., 'resnet'
parser.add_argument("-mn", "--multi_network", default='basic') 
parser.add_argument("-m", "--mode", default='train')  # 'train' or 'test'
parser.add_argument("-bs", "--batch_size", default=32)
parser.add_argument("-g", "--gpus", default="0,1,2,3") 
parser.add_argument("-e", "--earlystop", default=True)
parser.add_argument("-sl", "--schedule_limit", default=20)
parser.add_argument("--im_size", default=512)
parser.add_argument("--random_seed", default=20)
parser.add_argument("--class_outline", default=12)  # defines range for noncancerous labels

args = parser.parse_args()
dataset_name = args.dataroot
target_data = args.data
# For multimodal fusion, use UBF_VHE
if target_data == 'UBF_VHE':
    target_network = args.multi_network
else:
    target_network = args.network
mode = args.mode
class_outline = args.class_outline
earlystop_loss = args.earlystop
batch_size = int(args.batch_size)
gpus = args.gpus
img_size = int(args.im_size) 
random_seed = int(args.random_seed) 
scheduler_lim = int(args.schedule_limit)

prj_name = args.name
print(prj_name)
# Default settings
learning_rate = 0.0001
epochs = args.n_epochs
nfold_num = [0, 1, 2, 3, 4]
scheduler_metrics = 'loss'  # 'loss' or 'score'
cpu_workers = int(os.cpu_count() / 4)

tta_aug = []  # Optionally add augmentations like: tta.HorizontalFlip(), tta.VerticalFlip(), etc.
save_image = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setting Random Seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)
random.seed(random_seed)
def seed_worker(worker_id):
    np.random.seed(random_seed)
    random.seed(random_seed)

def to_var(tensor):
    return Variable(tensor.to(device))     

tap_char = '\t'

# Folder Setting
ckpt_dir = f'{args.checkpoints_dir}/{prj_name}'
result_dir = f'{args.results_dir}'
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# ---------------------------------------
# DATASET LOADER
# ---------------------------------------
class DatasetLoader(Dataset):
    def __init__(self, root, transform, tvt, didx=[], data_name='', ncan_list=[]):
        self.data_name = data_name
        self.image_list = []
        self.image_list2 = []  # used only in UBF_VHE mode
        if tvt != 'test':
            for d_idx in didx:
                if data_name != 'UBF_VHE':
                    self.image_list.extend(sorted(glob.glob(f'{root}/{data_name}/train/*_{d_idx:03d}_*.png')))
                else:
                    # In UBF_VHE mode, UBF and VHE images are stored in separate folders.
                    self.image_list.extend(sorted(glob.glob(f'{root}/UBF/train/*_{d_idx:03d}_*.png')))
                    self.image_list2.extend(sorted(glob.glob(f'{root}/VHE/train/*_{d_idx:03d}_*.png')))
        else:
            if data_name != 'UBF_VHE':
                self.image_list = sorted(glob.glob(f'{root}/{data_name}/test/*'))
            else:
                self.image_list = sorted(glob.glob(f'{root}/UBF/test/*'))
                self.image_list2 = sorted(glob.glob(f'{root}/VHE/test/*'))
        self.transform = transform
        self.tvt = tvt
        self.ncan_list = ncan_list
            
    def __len__(self):
        return len(self.image_list)
        
    def __getitem__(self, index):
        if self.data_name != 'UBF_VHE':
            image = Image.open(self.image_list[index])
            # For UBF and VHE, treat images as RGB.
            image = np.transpose(np.asarray(image), (2, 0, 1))
            image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                        torch.Tensor(image.copy()) / 255.0)
            imag = np.transpose(image, (1, 2, 0))
        else:
            # For UBF_VHE: load UBF and VHE images as RGB and concatenate along channel dimension.
            ubf_image = Image.open(self.image_list[index])
            ubf_image = np.transpose(np.asarray(ubf_image), (2, 0, 1))
            ubf_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                        torch.Tensor(ubf_image.copy()) / 255.0)
            
            vhe_image = Image.open(self.image_list2[index])
            vhe_image = np.transpose(np.asarray(vhe_image), (2, 0, 1))
            vhe_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                        torch.Tensor(vhe_image.copy()) / 255.0)
            
            # Concatenate along channel axis: result is a 6-channel image.
            image_cat = torch.cat([ubf_image, vhe_image], dim=0)
            imag = np.transpose(image_cat.cpu().numpy(), (1, 2, 0))
        
        imag = self.transform(np.array(imag)).float()
        
        # Determine label based on filename using 'class_outline'
        name = self.image_list[index].split('/')[-1]
        # Filename expected format: something_<ID>_<index>.png (e.g., img_012_003.png)
        if int(name.split('_')[-2]) in self.ncan_list:
            label = 0  # noncancerous
        else:
            label = 1  # cancerous
        
        return imag, label, index

# ---------------------------------------
# MODEL DEFINITION (Dual-branch uses separate 512→16 layers)
# ---------------------------------------
class MyModel(nn.Module):
    def __init__(self, data_name, model):
        super(MyModel, self).__init__()
        self.data_name = data_name

        # helper to build a ResNet18 backbone that outputs 512-D features
        def resnet18_backbone():
            m = timm.create_model('resnet18', in_chans=3, num_classes=2)
            # replace the final fc with identity → we get 512-D features
            m.fc = nn.Identity()
            return m

        if self.data_name == 'UBF_VHE':
            # dual branches
            self.backbone_ubf = resnet18_backbone()
            self.backbone_vhe = resnet18_backbone()
            # IMPORTANT: two separate 512→16 projections (one per branch)
            self.class512to16_ubf = nn.Linear(512, 16)
            self.class512to16_vhe = nn.Linear(512, 16)
            # concat(16 + 16 = 32) -> 2
            self.class32to2 = nn.Linear(32, 2)
        else:
            # single branch (UBF / VHE / HE all share the same head design)
            self.backbone_single = resnet18_backbone()
            self.class512to16_single = nn.Linear(512, 16)
            self.class16to2 = nn.Linear(16, 2)

    def forward(self, xinput):
        if self.data_name == 'UBF_VHE':
            # xinput: (B, 6, H, W) -> split into UBF (0:3) & VHE (3:6)
            f1 = self.backbone_ubf(xinput[:, 0:3, :, :])   # (B, 512)
            f2 = self.backbone_vhe(xinput[:, 3:6, :, :])   # (B, 512)
            z1 = self.class512to16_ubf(f1)                 # (B, 16)
            z2 = self.class512to16_vhe(f2)                 # (B, 16)
            f  = torch.cat([z1, z2], dim=1)                # (B, 32)
            out = self.class32to2(f)                       # (B, 2)
        else:
            # single-input path (UBF/VHE/HE): 512 -> 16 -> 2
            f   = self.backbone_single(xinput)             # (B, 512)
            z   = self.class512to16_single(f)              # (B, 16)
            out = self.class16to2(z)                       # (B, 2)
        return out

# ---------------------------------------
# TRAINING FUNCTIONS
# ---------------------------------------
def train_(net, train_dataset, valid_dataset, optimizer, fold, data_name, model):
    criterion = FocalLoss(gamma=2).cuda() 
    best_epoch = 0
    score_max = -1.0
    loss_min = 1e8
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = 0
        net.train()
        for input, label, _ in train_dataset:
            input = input.to(device)
            label = label.to(device) 
            optimizer.zero_grad()
            predicts = net(input)
            loss_train = criterion(predicts.float(), label)
            train_loss += loss_train
            loss_train.backward()
            optimizer.step()
        current_score, valid_loss = evaluation(net, valid_dataset, criterion)
        train_loss /= len(train_dataset)

        if (valid_loss < loss_min and earlystop_loss) or (current_score > score_max and not earlystop_loss):
            best_epoch = epoch
            loss_min = valid_loss
            score_max = current_score
            save_path = f'{ckpt_dir}/{data_name}/{model}_fold{fold}_best.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(net.state_dict(), save_path)

        if epoch - best_epoch > scheduler_lim:
            break

    return best_epoch, score_max  

def evaluation(net, valid_dataset, criterion):
    valid_loss = 0
    with torch.no_grad():
        net.eval()
        accuracy = 0
        for input, label, _ in valid_dataset:
            input = input.to(device)
            label = label.to(device) 
            predicts = net(input)
            loss_valid = criterion(predicts.float(), label)
            valid_loss += float(loss_valid)
            ps = torch.exp(predicts)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == label.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        out = accuracy / len(valid_dataset)
    return out, valid_loss / len(valid_dataset)

def test_(net, test_dataset, tot_num):
    list_tot = np.zeros((tot_num))
    list_lab1 = np.zeros((tot_num))
    list_real = np.zeros((tot_num))

    transforms_tta = tta.Compose(tta_aug)
    for transformer in transforms_tta:
        with torch.no_grad():
            net.eval()
            for input, label, idx2 in test_dataset:
                input = transformer.augment_image(input)
                input = input.to(device)
                label = label.to(device)
                logps = net(input)
                ps = torch.exp(logps)
                _, top_class = ps.topk(1, dim=1)
                # (Adjust if your idx2 represents a batch index.)
                for ix in range(len(top_class)):
                    list_tot[idx2 + ix] += 1 
                    if top_class[ix] == 1: 
                        list_lab1[idx2 + ix] += 1
                    if label.view(*top_class.shape)[ix] == 1: 
                        list_real[idx2 + ix] = 1
    test_out = []
    for ix in range(tot_num):
        an = 1 if list_lab1[ix] >= (list_tot[ix] - list_lab1[ix]) else 0
        test_out.append(an)    
    return test_out, list_tot, list_lab1, list_real

# ---------------------------------------
# MAIN SCRIPT
# ---------------------------------------
# Set noncancerous/cancerous range based on class_outline.
ncan_list = [i for i in range(class_outline)]
if __name__ == "__main__":
    print(f'\nInfo : Dataset {target_data}')
    
    if target_data != 'UBF_VHE':
        data_dir_tmp = f'{dataset_name}/{target_data}'
    else:
        data_dir_tmp = f'{dataset_name}/UBF'
    tr_list = sorted(glob.glob(f'{data_dir_tmp}/train/*.png'))
    ts_list = sorted(glob.glob(f'{data_dir_tmp}/test/*.png'))
    test_size = len(ts_list)
    ncan_set = set()
    can_set = set()
    for tr in tr_list:
        idx = int(tr.split('/')[-1].split('_')[-2])
        if idx in ncan_list:
            ncan_set.add(idx)
        else:
            can_set.add(idx)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True)
    train_ncan_list, valid_ncan_list = [], []
    for train, test in kf.split(list(ncan_set)):
        train_ncan_list.append([list(ncan_set)[idx] for idx in train])
        valid_ncan_list.append([list(ncan_set)[idx] for idx in test])
    train_can_list, valid_can_list = [], []
    for train, test in kf.split(list(can_set)):
        train_can_list.append([list(can_set)[idx] for idx in train])
        valid_can_list.append([list(can_set)[idx] for idx in test])

    print(target_network)
    if mode == 'train':
        for fold in nfold_num:
            print(f'Fold {fold}, Train/Valid split')
            train_index = []
            train_index.extend(train_ncan_list[fold])
            train_index.extend(train_can_list[fold])
            valid_index = []
            valid_index.extend(valid_ncan_list[fold])
            valid_index.extend(valid_can_list[fold])
            print(train_index)
            print(valid_index)
            # Build Model
            net = MyModel(target_data, target_network)
            if torch.cuda.device_count() > 1:
                net = torch.nn.DataParallel(net).to(device)
            else:
                net = net.to(device)

            # Data Preparation
            transform_t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]) 
            transform_v = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size))
            ])
            print(f'{dataset_name}')
            train_data = DatasetLoader(f'{dataset_name}', transform_t, 'train',
                                       didx=train_index, data_name=target_data,
                                       ncan_list=ncan_list)
            valid_data = DatasetLoader(f'{dataset_name}', transform_v, 'valid',
                                       didx=valid_index, data_name=target_data,
                                       ncan_list=ncan_list)
            train_dataset = DataLoader(dataset=train_data, batch_size=batch_size,
                                       shuffle=True, num_workers=cpu_workers, worker_init_fn=seed_worker)
            valid_dataset = DataLoader(dataset=valid_data, batch_size=batch_size,
                                       shuffle=True, num_workers=cpu_workers, worker_init_fn=seed_worker)

            if fold == 0:
                print('Dataset (number of batches x batch size)')
                print('num_train : %d x %d' % (len(train_dataset), batch_size))
                print('num_valid : %d x %d' % (len(valid_dataset), batch_size))
            
            optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
            best_epoch, score_max = train_(net, train_dataset, valid_dataset, optimizer, fold, target_data, target_network)
            
    end = time.time()
    print(f"TRAININGTIME {end - start:.5f} sec")
    
    # TESTING PHASE
    start = time.time()
    net = MyModel(target_data, target_network)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).to(device)
    else:
        net = net.to(device)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size))
    ])
    test_data = DatasetLoader(f'{dataset_name}', transform, 'test', data_name=target_data)
    test_dataset = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                              num_workers=cpu_workers, worker_init_fn=seed_worker)

    print('\nDataset (number of batches x batch size)')
    print('num_test : %d x %d' % (len(test_dataset), batch_size))

    result_list = []
    for fold in range(5):
        # ---- FIXED: load from the same subdir used when saving ----
        saved_model_path = f'{ckpt_dir}/{target_data}/{target_network}_fold{fold}_best.pth'
        net.load_state_dict(torch.load(saved_model_path, map_location=device))
        result_list.append(test_(net, test_dataset, test_size))
        
    f = open(f'{result_dir}/{target_network}_result_history.csv', 'w')
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    for fd in range(5):
        y_pred = []
        y_true = []
        for ix in range(test_size):
            y_pred.append(result_list[fd][2][ix])
            y_true.append(result_list[fd][3][ix])
        cm = confusion_matrix(y_true, y_pred)
        acc_list.append(accuracy_score(y_true, y_pred))
        f1_list.append(f1_score(y_true, y_pred))
        recall_list.append(recall_score(y_true, y_pred))
        precision_list.append(precision_score(y_true, y_pred))
    print(f'acc {np.mean(acc_list)}+-{np.std(acc_list)}')
    print(f'f1 {np.mean(f1_list)}+-{np.std(f1_list)}')
    print(f'recall {np.mean(recall_list)}+-{np.std(recall_list)}')
    print(f'precision {np.mean(precision_list)}+-{np.std(precision_list)}')
    
    f.write('acc, f1, precision, recall\n')
    for i in range(5):
        f.write(f'{acc_list[i]}, {f1_list[i]}, {precision_list[i]}, {recall_list[i]}\n')
    f.write('mean\n')
    f.write(f'acc {np.mean(acc_list)}+-{np.std(acc_list)}\n')
    f.write(f'f1 {np.mean(f1_list)}+-{np.std(f1_list)}\n')
    f.write(f'precision {np.mean(precision_list)}+-{np.std(precision_list)}\n')
    f.write(f'recall {np.mean(recall_list)}+-{np.std(recall_list)}\n')
    f.close()

print('Wait for end')
end = time.time()
print(f"TESTINGTIME {end - start:.5f} sec")
sleep(3)
