import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from faceforensics2 import FaceForensics
import yaml
import config
import os
import sys
import time
import math
import yaml
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
#import pytorch_warmup as warmup
from celeb_df import CelebDF

from tqdm import tqdm
from pprint import pprint
from torch.utils import data
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
#from tensorboardX import SummaryWriter
import pdb
import torch.nn as nn
#import wandb
from torch.cuda import amp
import torch.distributed as dist
#from tes import SmoothCrossEntropy
from datetime import datetime
#from am_softmax import AMSoftmaxLoss
import torch.nn.functional as F
class GeneralizeCrossEntropy(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizeCrossEntropy, self).__init__()
        self.q = q

    def forward(self, logits, labels):
        # Negative box cox: (1-f(x)^q)/q
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
        probs = F.softmax(logits, dim=-1)
        loss = (1 - torch.pow(torch.sum(labels * probs, dim=-1), self.q)) / self.q
        loss = torch.mean(loss)
        return loss


loass_a=GeneralizeCrossEntropy()
from utils2 import AUCMeter,AccMeter,AverageMeter
LOADERS = {
    "FaceForensics": FaceForensics,
    "CelebDF": CelebDF,
}

def load_dataset(name="FaceForensics"):
    print(f"Loading dataset: '{name}'...")
    return LOADERS[name]

with open("config/Recce.yml") as config_file:
        data_cfg = yaml.load(config_file, Loader=yaml.FullLoader)
#data_cfg=configs.get("model", None)
#print(data_cfg)
dist.init_process_group("nccl")

train_dataset = data_cfg["data"]["file"]
branch = data_cfg["data"]["train_branch"]
name = data_cfg["data"]["name"]
with open(train_dataset, "r") as f:
    options = yaml.load(f, Loader=yaml.FullLoader)

val_branch = data_cfg["data"]["val_branch"]

# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype
#
#     def forward(self, prompts, tokenized_prompts):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)
#
#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
#         return x

#print(options)

# sweep_config = {
#     'method': 'random',
#     'name': 'sweep',
#     'metric': {
#         'goal': 'minimize',
#         'name': 'cur_acc'
#         },
#     'parameters': {
#         'batch_size': {'values': [125, 256, 512,1024]},
#         'epochs': {'values': [5, 10, 15,20,25,30,40,50,60]},
#         'lr': {'max': 1e-3, 'min': 1e-8}
#      }
# }
# sweep_id = wandb.sweep(sweep_config, project="wandb_demo")
# wandb.init()

from torch.cuda.amp import autocast as autocast
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device(f'cuda:{1}' if torch.cuda.is_available() else 'cpu')

#device = "cuda:1" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load('ViT-B/32',device=device,jit=False) #Must set jit=False for training
print(preprocess)

def freeze(m):
    for i, k in m.named_children():
        try:
            freeze(k)
            if isinstance(k,nn.BatchNorm2d):
                k.eval()
        except:
            raise NotImplementedError
        
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


train_options = options[branch]
train_set = load_dataset(name)(train_options, preprocess,mode="train")
train_sampler = data.distributed.DistributedSampler(train_set)

#val_options = options[val_branch]# 这里没有用val branch的option
val_options = options[val_branch]
val_set = load_dataset(name)(val_options, preprocess,mode="val")
val_sampler = data.distributed.DistributedSampler(val_set)
val_dataloader = data.DataLoader(val_set, shuffle=False,
                                            sampler=val_sampler,
                                            num_workers=data_cfg.get("num_workers", 0),
                                            batch_size=16)

#val_options = options[data_cfg["val_branch"]]
# val_set = load_dataset(name)(val_options, preprocess)
#             # wrapped with data loader
# val_loader = data.DataLoader(val_set, shuffle=True,
#                                               num_workers=data_cfg.get("num_workers", 4),
#                                               batch_size=data_cfg["val_batch_size"])

# class image_title_dataset(Dataset):
#     def __init__(self, list_image_path,list_txt):

#         self.image_path = list_image_path
#         self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

#     def __len__(self):
#         return len(self.title)

#     def __getitem__(self, idx):
#         image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
#         title = self.title[idx]
#         return image,title

# # use your own data
# list_image_path = ['folder/image1.jpg','folder2/image2.jpg']
# list_txt = ['description for image1.jpg' , 'description for image2.jpg']
# dataset = image_title_dataset(list_image_path,list_txt)
# train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE) #Define your own dataloader


# #https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.float()
        p.grad.data = p.grad.float()

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
input_size = 512  # 输入特征的维度
hidden_size = 512  # 隐层的大小
num_classes = 5 # 分类的类别数
mlps = MLP(input_size, hidden_size, num_classes).to("cuda:1")
# if device == "cpu":
#   model.float()
# else :
#   clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
EPOCH=40

max_epoch = EPOCH





# add your own code to track the training progress.
text_mapping = {
    0: "a photo of real face",
    1: "a photo of fake face"
}
loss_criterion=nn.CrossEntropyLoss()

val_acccs=0
def CrossEntropyLoss_label_smooth(outputs, targets,
                                  num_classes=2, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes),
                                 fill_value=epsilon / (num_classes - 1)).to("cuda:1")
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
                             value=1-epsilon)
    log_prob = nn.functional.log_softmax(outputs, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss

#cls_criterion = AMSoftmaxLoss(gamma=0., m=0.45, s=30, t=1.)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=1e-6) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6,betas=(0.9,0.98),eps=1e-4,weight_decay=1e-6)
#optimizer= torch.optim.SGD(model.parameters(), lr=1e-3, momentum=1e-6)

train_dataloader = data.DataLoader(train_set, shuffle=False,
                                            sampler=train_sampler,
                                            num_workers=data_cfg.get("num_workers", 1),
                                            batch_size=32)
lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=0)
# num_steps = len(train_dataloader) * max_epoch
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
# warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

#lr_scheduler = ReduceLROnPlateau(optimizer, 'max',factor=0.5, patience=4, verbose=True)
# def freeze_bn(self):
#     for m in self.network.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.eval()
scaler=amp.GradScaler()
step=0
#Loss = SmoothCrossEntropy()
f1 = open("training_log.csv", 'a+')
model=model.float()
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


#model=convert_models_to_fp32(model)
import torch.nn.functional as F
def norm_n_corr(x):
    norm_embed = F.normalize(x, p=2, dim=1)
    corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
    return corr
class MLLoss(nn.Module):
    def __init__(self):
        super(MLLoss, self).__init__()

    def forward(self, input, target, eps=1e-6):
        # 0 - real; 1 - fake.
        loss = torch.tensor(0., device=target.device)
        batch_size = target.shape[0]
        mat_1 = torch.hstack([target.unsqueeze(-1)] * batch_size)
        mat_2 = torch.vstack([target] * batch_size)
        diff_mat = torch.logical_xor(mat_1, mat_2).float()
        or_mat = torch.logical_or(mat_1, mat_2)
        eye = torch.eye(batch_size, device=target.device)
        or_mat = torch.logical_or(or_mat, eye).float()
        sim_mat = 1. - or_mat
        for _ in input:
            diff = torch.sum(_ * diff_mat, dim=[0, 1]) / (torch.sum(diff_mat, dim=[0, 1]) + eps)
            sim = torch.sum(_ * sim_mat, dim=[0, 1]) / (torch.sum(sim_mat, dim=[0, 1]) + eps)
            partial_loss = 1. - sim + diff
            loss += max(partial_loss, torch.zeros_like(partial_loss))
        return loss


contra_loss = MLLoss()

for epoch in range(EPOCH):
    print("now is "+str(epoch+1))
    total_loss = 0
    model.train()
    #freeze_bn(model)
    #freeze(model)
    for batch in tqdm(train_dataloader, total=len(train_dataloader)) :
        I, Y,te,types= batch
#         import pdb
#         pdb.set_trace()
        optimizer.zero_grad()
        images= I.to(device)
        types= types.to(device)
        Y=Y.to(device)
        texts=te.squeeze(1).to(device)
        # with amp.autocast(enabled=True):
        #image_features = model.encode_image(images)
        #image_featuresa=norm_n_corr(image_features)
        #image_features=mlps(image_features)
        #type_loss=loss_img(image_features,types)
        # import pdb
        # pdb.set_trace()
        #kl = F.kl_div(image_features.softmax(dim=-1).log(), types.float().softmax(dim=-1), reduction='sum')
       # c_loss=contra_loss(image_featuresa, Y)
        logits_per_image, logits_per_text = model(images, texts)
        # import pdb 
        # pdb.set_trace()
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss2 = (loss_img(logits_per_image,ground_truth) + loss_img(logits_per_text,ground_truth))/2#+type_loss
        total_loss+=total_loss2.item()
        total_loss2.backward()
        optimizer.step()
        #lr_scheduler.step()
        # with warmup_scheduler.dampening():
        #     lr_scheduler.step()
        # import pdb
        # pdb.set_trace()
        # scaler.scale(total_loss2.to(torch.float16)).backward()
        # #scaler.step(optimizer)
        #optimizer.step()
        #scaler.update()
    total_loss = total_loss / len(train_dataloader)
    #lr_scheduler.step()
    lr_scheduler.step()
    #warmup_scheduler.dampen()

    ##scale.step(lr_scheduler)
    with torch.no_grad():
        acc = AccMeter()
        auc = AUCMeter()
        loss_meter = AverageMeter()
        cur_acc = 0.0  # Higher is better
        cur_auc = 0.0  # Higher is better
        cur_loss = 1e8  # Lower is better
        model.eval()
        now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        for batch in val_dataloader:
            I, Y, te,types = batch
            images = I.to(device)
            Y = Y.to(device)

            texts = val_dataloader.dataset.get_text()
            texts = texts.to(device)
           # logits_per_image, logits_per_text = model(images, texts)
           # image_features = model.encode_image(0)

            #probs=mlps(image_features)
            logits_per_image, logits_per_text = model(images, texts)

            probs = logits_per_image.softmax(dim=-1)  # .cpu().numpy()

            acc.update(probs, Y, False)
            auc.update(probs, Y, False)
            cur_acc = acc.mean_acc()
            # import pdb
            # pdb.set_trace()
            #print(auc.mean_auc())
        cur_auc = auc.mean_auc()
        # torch.save({
        # 'epoch': epoch,
        # 'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': total_loss,
        # }, f"model_checkpoint/model_"+str(epoch)+".pt") #just change to your preferred folder/filename
        # print(cur_acc)
        val_msg = '[%s] Epoch: %d |  AUC: %f | ACC: %f|' % (
                    now_time, epoch + 1, cur_auc, cur_acc)
        print('\n', val_msg)
        with open('celebdf_clip.txt', 'a', encoding = 'utf-8') as f: 
            f.write(val_msg)
            f.write('\n')
        f1.write(val_msg)
        f1.write('\n')
        print("Eval Epoch %d, Loss %.4f, ACC %.4f, AUC %.4f" % (epoch+1, total_loss, cur_acc, cur_auc))
        #wandb.log({"acc": cur_acc, "cur_auc": cur_auc})
    #   #I = train_loader.dataset.load_item(I)
f1.close()

     # pdb.set_trace()

    #   logits_per_image, logits_per_text = model(images, texts)

    #   ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

    #   total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
    #   total_loss.backward()
    #   if device == "cpu":
    #      optimizer.step()
    #   else :
    #     convert_models_to_fp32(model)
    #     optimizer.step()
    #     clip.model.convert_weights(model)