from __future__ import print_function, division
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from opt.dataloader2 import face_Dataset, my_transforms
from arch.xception import xception
#from tqdm import tqdm
from utils.metrics_intra import get_metrics
import random
import pandas as pd

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
criterion = torch.nn.BCEWithLogitsLoss()
import json
from glob import glob
from os.path import join
from dataset import AbstractDataset
from init import *
import yaml
from torch.utils.data import DataLoader, ConcatDataset
from utils2 import AUCMeter,AccMeter,AverageMeter

import cv2
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
#tqdm模块错误
config_path = "config/dataset/celeb_df.yml"
with open(config_path) as config_file:
    config_celeb = yaml.load(config_file, Loader=yaml.FullLoader)
celeb_config = config_celeb["train_cfg"]
val_celeb_config= config_celeb["test_cfg"]

wild_config_path = "config/dataset/wilddeepfake.yml"
with open(wild_config_path) as config_file:
    config_wild_wild = yaml.load(config_file, Loader=yaml.FullLoader)
wild_config = config_wild_wild["train_cfg"]
val_wild_config= config_wild_wild["test_cfg"]



dfdc_config_path = "config/dataset/dfdc.yml"
with open(dfdc_config_path) as config_file:
    config_dfdc = yaml.load(config_file, Loader=yaml.FullLoader)
#wild_config = config_dfdc["train_cfg"]
val_dfdc_config= config_dfdc["test_cfg"]
val_dfdc_dataset = DFDC(val_dfdc_config)

celeb_dataset = CelebDF(celeb_config)
wild_dataset = WildDeepfake(wild_config)

val_celeb_dataset = CelebDF(val_celeb_config)

# import pdb
# pdb.set_trace()

#combined_dataset = ConcatDataset([celeb_dataset, wild_dataset])

# def custom_collate(batch):
#     filtered_batch = []
#     for video,  label in batch:
#         filtered_batch.append((video, label))
#     return torch.utils.data.dataloader.default_collate(filtered_batch)

config_path = "config/dataset/faceforensics.yml"
with open(config_path) as config_file:
    config_40 = yaml.load(config_file, Loader=yaml.FullLoader)
c40_config = config_40["train_cfg"]
val_c40_config= config_40["test_cfg"]

c23_config_path = "config/dataset/faceforensics23.yml"
with open(c23_config_path) as config_file:
    config_23 = yaml.load(config_file, Loader=yaml.FullLoader)
c23_config = config_23["train_cfg"]
val_c23_config= config_23["test_cfg"]

val_c23_dataset = FaceForensics(c23_config)


c40_dataset = FaceForensics(c40_config)
c23_dataset = FaceForensics(c23_config)


val_c40_dataset = FaceForensics(val_c40_config)

#combined_dataset3 = ConcatDataset([c40_dataset, c23_dataset])

#c23 celeb
#combined_dataset2 = ConcatDataset([wild_dataset, combined_dataset3])
combined_dataset=ConcatDataset([c40_dataset, wild_dataset])

# batch_size = 32
# train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_c40_dataset, batch_size=batch_size, shuffle=True)


f1 = open("training_log.csv", 'a+')


def parse_args():
    parser = argparse.ArgumentParser(description='cq')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--warm_start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_val', default=64, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--save_model', default=3000, type=int)
    parser.add_argument('--disp_step', default=500, type=int)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--save_root', default='./Training_results/Xception_baseline/', type=str)
    parser.add_argument('--train_csv', default='./misc_intra/Train/train.csv', type=str)
    parser.add_argument('--val_csv', default='./misc_intra/Val100/val.csv', type=str)
    parser.add_argument('--root_path', default='', type=str)
    parser.add_argument('--model_name', default="Xcepbaseline_c40_wild/", type=str)
    return parser.parse_args()

def fix_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validation(model, dataloader, num_imgs, thre):
    print('Validating...')
    model.eval()
    batch_val_losses = []
    SCORE = np.zeros(num_imgs)
    LABEL = np.zeros(num_imgs)
    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            acc = AccMeter()
            auc = AUCMeter()
            loss_meter = AverageMeter()
            cur_acc = 0.0  # Higher is better
            cur_auc = 0.0  # Higher is better
            cur_loss = 1e8  # Lower is better
#             import pdb
#             pdb.set_trace()
            faces,label_cls=data
            faces=faces.to(device)
        
           # faces=load_item(faces).to(device)
            label_cls=label_cls.to(device)
            Y=label_cls
            model=model.to(device)

            x = model(faces)
            probs = torch.squeeze(torch.sigmoid(x), 1)
            acc.update(probs, Y, True)
            auc.update(probs, Y, True)
            cur_acc = acc.mean_acc()
    cur_auc = auc.mean_auc()
    val_msg = '[%s] Epoch: %d |  AUC: %f | ACC: %f|' % (
                    0, 0 + 1, cur_auc, cur_acc)
    print('\n', val_msg)
    with open('c23_c40_valdfdc.txt', 'a', encoding = 'utf-8') as f: 
        f.write(val_msg)
        f.write('\n')
        f1.write(val_msg)
    f1.write('\n')
    print("Eval Epoch %d, Loss %.4f, ACC %.4f, AUC %.4f" % (0+1, 0, cur_acc, cur_auc))

    return 0, 0, 0

def test(args, model):
    training_dataloader = DataLoader(combined_dataset, batch_size=args.batch_size_train, shuffle=True)
    val_dataloader = DataLoader(val_dfdc_dataset, batch_size=args.batch_size_val, shuffle=True,drop_last=True)
    val_loss, AUC, ACC = validation(model, val_dataloader, 179000, 0.5)
    

def train(args, model):
    avg_train_loss_list = np.array([])
    training_dataloader = DataLoader(combined_dataset, batch_size=args.batch_size_train, shuffle=True)

    val_dataloader = DataLoader(val_c23_dataset, batch_size=args.batch_size_val, shuffle=True,drop_last=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Num_val_imgs=179000

    # result folder
    res_folder_name = args.save_root + args.model_name
    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)
        os.mkdir(res_folder_name + '/ckpt/')
    else:
        print("WARNING: RESULT PATH ALREADY EXISTED -> " + res_folder_name)
    print('find models here: ', res_folder_name)
    writer = SummaryWriter(res_folder_name)
    f1 = open(res_folder_name + "/training_log.csv", 'a+')
    Best_AUC = 0
    # training
    steps_per_epoch = len(training_dataloader)
    for epoch in range(args.warm_start_epoch, args.epochs):
        batch_train_losses = []
        step_loss = np.zeros(steps_per_epoch, dtype=np.float)
        for step,data in enumerate(tqdm(training_dataloader)):
            model.train()
            optimizer.zero_grad()
            frames,labels=data
            frames=frames.to(device)
            labels=labels.to(device)
            model=model.to(device)

            #frames=load_item(frames).to(device)
            
            predicted_label =torch.squeeze(model(frames), 1)
            loss = criterion(predicted_label, labels.float())
            step_loss[step] = loss
            batch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            Global_step = epoch * steps_per_epoch + (step + 1)

            if Global_step % args.disp_step == 0:
                avg_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
                now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                step_log_msg = '[%s] Epoch: %d/%d | Global_step: %d |average loss: %f' % (
                now_time, epoch + 1, args.epochs, Global_step, avg_loss)
                writer.add_scalar('Loss/train', avg_loss, Global_step)
                print('\n', step_log_msg)

            if Global_step % args.save_model == 0:
                now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                avg_train_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
                
                avg_train_loss_list = np.append(avg_train_loss_list, avg_train_loss)
                log_msg = '[%s] Epoch: %d/%d | 1/10 average epoch loss: %f' % (now_time, epoch + 1, args.epochs, avg_train_loss)
                print('\n', log_msg)
                f1.write(log_msg)
                f1.write('\n')

                # validation
                val_loss, AUC, ACC = validation(model, val_dataloader, Num_val_imgs, 0.5)
                val_msg = '[%s] Epoch: %d/%d | Global_step: %d | average validation loss: %f | ACC: %f| AUC: %f' % (now_time, epoch + 1, args.epochs, Global_step, val_loss, ACC, AUC)
                print('\n', val_msg)
                f1.write(val_msg)
                f1.write('\n')

                # save model
                if AUC > Best_AUC and Global_step>30000:
                    Best_AUC = AUC
                    torch.save(model.state_dict(), res_folder_name + '/ckpt/' + 'best.pth')
                    np.save(res_folder_name + '/avg_train_loss_list.np', avg_train_loss_list)
                    cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
                    print('Saved model. lr %f' % cur_learning_rate[0])
                    f1.write('Saved model. lr %f' % cur_learning_rate[0])
                    f1.write('\n')
    f1.close()


def main(args):
    model = xception(pretrained=True)
    
    model = model.to(device)
    print(model)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    #train(args, model)
    model.load_state_dict(torch.load("Training_results/Xception_baseline_c40_c23/Xcepbaseline/ckpt/best.pth"))
    
    test(args,model)

if __name__ == '__main__':
    args = parse_args()
    if args.random_seed is not None:
        fix_seed(args.random_seed)
    print(args)
    main(args)
