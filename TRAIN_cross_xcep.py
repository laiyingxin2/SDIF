from __future__ import print_function, division
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from opt.dataloader import face_Dataset, my_transforms
from arch.xception import xception
from tqdm import tqdm
import random
import pandas as pd
from utils.metrics_cross import get_metrics
import torch.nn as nn



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
criterion = torch.nn.BCEWithLogitsLoss()
adv_fuc = nn.CrossEntropyLoss()
import json
from glob import glob
from os.path import join
from dataset import AbstractDataset
from init import *
import yaml
from torch.utils.data import DataLoader, ConcatDataset

import cv2
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
 

config_path = "config/dataset/celeb_df.yml"
with open(config_path) as config_file:
    config_celeb = yaml.load(config_file, Loader=yaml.FullLoader)
celeb_config = config_celeb["train_cfg"]
val_celeb_config= config_celeb["test_cfg"]

wild_config_path = "config/dataset/wilddeepfake.yml"
with open(wild_config_path) as config_file:
    config_wild = yaml.load(config_file, Loader=yaml.FullLoader)
wild_config = config_wild["train_cfg"]
val_wild_config= config_wild["test_cfg"]

celeb_dataset = CelebDF(celeb_config)
wild_dataset = WildDeepfake(wild_config)

val_celeb_dataset = CelebDF(val_celeb_config)



#combined_dataset = ConcatDataset([celeb_dataset, wild_dataset])

#batch_size = 32
# combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

# val_dataloader = DataLoader(val_celeb_dataset, batch_size=batch_size, shuffle=True)

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

c40_dataset = FaceForensics(c40_config)
c23_dataset = FaceForensics(c23_config)

val_c40_dataset = FaceForensics(val_c40_config)

#combined_dataset2 = ConcatDataset([c23_dataset])

#combined_dataset3 = ConcatDataset([combined_dataset2, celeb_dataset])

combined_dataset =c40_dataset# ConcatDataset([wild_dataset, combined_dataset3])
def parse_args():
    parser = argparse.ArgumentParser(description='cq')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_decay_rate', default=1.0, type=float)
    parser.add_argument('--lr_decay_step', default=999999999, type=str)
    parser.add_argument('--warm_start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_val', default=64, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--save_model', default=3000, type=int)
    parser.add_argument('--disp_step', default=500, type=int)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--save_root', default='./Training_results——TTTc40/Cross_df/', type=str)
    parser.add_argument('--root_path', default='', type=str)
    parser.add_argument('--model_name', default="Xception/", type=str)
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

def val(model, dataloader, num_imgs):
    print('Validating...')
    model.eval()
    batch_val_losses = []
    SCORE = np.zeros(num_imgs)
    LABEL = np.zeros(num_imgs)
    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            faces,label_cls=data
            faces=faces.to(device)
           # faces=load_item(faces).to(device)
            label_cls=label_cls.to(device)
            x = model(faces)
            pred_score = torch.squeeze(torch.sigmoid(x), 1)
            val_loss = criterion(torch.squeeze(x, 1), label_cls.float())
            batch_val_losses.append(val_loss.item())
            SCORE[num * args.batch_size_val:(num + 1) * args.batch_size_val] = pred_score.cpu().numpy()
            LABEL[num * args.batch_size_val:(num + 1) * args.batch_size_val] = label_cls.cpu().numpy()
    avg_val_loss = round(sum(batch_val_losses) / (len(batch_val_losses)), 5)
    pred = SCORE[:, np.newaxis]
    y_true = LABEL[:, np.newaxis]
    return avg_val_loss, pred, y_true


def combine_csv(file_array, result_name):
    combine_cmd = "cat "
    for i in range(len(file_array)):
        combine_cmd += (file_array[i] + " ")
    combine_cmd += " > " + result_name
    os.system(combine_cmd)


def train(args, model):
    avg_train_loss_list = np.array([])
    training_dataloader = DataLoader(c23_dataset, batch_size=args.batch_size_train, shuffle=True)
    val_dataloader = DataLoader(val_c40_dataset, batch_size=args.batch_size_val, shuffle=True,drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Num_val_all = 179000#Num_val_imgs_1 + Num_val_imgs_2 + Num_val_imgs_3 + Num_val_imgs_4
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    UUID = 1
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
    Learning_Rate = args.lr
    for epoch in range(args.warm_start_epoch, args.epochs):
        batch_train_losses = []
        step_loss = np.zeros(steps_per_epoch, dtype=np.float)
        for step, data in enumerate(tqdm(training_dataloader)):
            model.train()
            optimizer.zero_grad()
            frames,labels=data
            frames=frames.to(device)
            labels=labels.to(device)   
            predict,domain_invariant=model(frames)
            predicted_label = torch.squeeze(model(frames), 1)
            adv_loss =  adv_fuc(domain_invariant, UUID.long())
            loss = criterion(predicted_label, labels.float())+adv_loss
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
                log_msg = '[%s] Epoch: %d/%d | 1/10 average epoch loss: %f' % (
                    now_time, epoch + 1, args.epochs, avg_train_loss)
                print('\n', log_msg)
                f1.write(log_msg)
                f1.write('\n')

                # validation
                val_loss, pred, y = val(model, val_dataloader, Num_val_all)
                threshold = 0.5
                AUC, ACC, FPR, FNR, EER, AP = get_metrics(pred, y, threshold)

                val_msg = '[%s] Epoch: %d/%d | Global_step: %d | average df validation loss: %f | AUC: %f | ACC: %f| FPR: %f| FTR: %f| EER: %f| AP: %f' % (
                    now_time, epoch + 1, args.epochs, Global_step, val_loss, AUC, ACC, FPR, FNR, EER, AP)
                print('\n', val_msg)
                f1.write(val_msg)
                f1.write('\n')

                # save model
                if AUC > Best_AUC and Global_step > 50000:
                    Best_AUC = AUC
                    torch.save(model.state_dict(), res_folder_name + '/ckpt/' + 'best.pth')
                    np.save(res_folder_name + '/avg_train_loss_list.np', avg_train_loss_list)
                    cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
                    print('Saved model. lr %f' % cur_learning_rate[0])
                    f1.write('Saved model. lr %f' % cur_learning_rate[0])
                    f1.write('\n')

            if Global_step % args.lr_decay_step == 0:
                Learning_Rate = Learning_Rate * args.lr_decay_rate
                optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate, weight_decay=args.weight_decay)
    f1.close()


def main(args):
    model = xception(pretrained=True)
    model = model.to(device)
    print(model)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    train(args, model)

if __name__ == '__main__':
    args = parse_args()
    if args.random_seed is not None:
        fix_seed(args.random_seed)
    print(args)
    main(args)
