import cv2
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import torch.distributed as dist
import yaml
from init import *
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader

all_data=["config/dataset/faceforensics.yml","config/dataset/celeb_df.yml","config/dataset/dfdc.yml","config/dataset/wilddeepfake.yml"]
all_name=["FaceForensics","CelebDF","DFDC","WildDeepfake"]

dist.init_process_group("nccl")


with open("config/Recce.yml") as config_file:
        data_cfg = yaml.load(config_file, Loader=yaml.FullLoader)
        
train_dataset =all_data[0]
branch =data_cfg["data"]["train_branch"]
name =all_name[0]
with open(train_dataset, "r") as f:
    options = yaml.load(f, Loader=yaml.FullLoader)
train_options = options[branch]
train_set = load_dataset(name)(train_options)
face_sampler = data.distributed.DistributedSampler(train_set)
face_dataloader = data.DataLoader(train_set, shuffle=False,
                                            sampler=face_sampler,
                                            num_workers=data_cfg.get("num_workers", 4),
                                            batch_size=16)


val_branch = data_cfg["data"]["val_branch"]
val_options = options[val_branch]
val_set = load_dataset(name)(val_options)
face_val_sampler = data.distributed.DistributedSampler(val_set)
face_val_dataloader = data.DataLoader(val_set, shuffle=False,
                                            sampler=face_val_sampler,
                                            num_workers=data_cfg.get("num_workers", 4),
                                            batch_size=16)

train_dataset =all_data[1]
branch =data_cfg["data"]["train_branch"]
name =all_name[1]
with open(train_dataset, "r") as f:
    options = yaml.load(f, Loader=yaml.FullLoader)
train_options = options[branch]
train_set = load_dataset(name)(train_options)
celeb_sampler = data.distributed.DistributedSampler(train_set)
celeb_dataloader = data.DataLoader(train_set, shuffle=False,
                                            sampler=celeb_sampler,
                                            num_workers=data_cfg.get("num_workers", 4),
                                            batch_size=16)
val_branch = data_cfg["data"]["val_branch"]
val_options = options[val_branch]
val_set = load_dataset(name)(val_options)
celeb_val_sampler = data.distributed.DistributedSampler(val_set)
celeb_val_dataloader = data.DataLoader(val_set, shuffle=False,
                                            sampler=celeb_val_sampler,
                                            num_workers=data_cfg.get("num_workers", 4),
                                            batch_size=16)


train_dataset =all_data[3]
branch =data_cfg["data"]["train_branch"]
name =all_name[3]
with open(train_dataset, "r") as f:
    options = yaml.load(f, Loader=yaml.FullLoader)
train_options = options[branch]
train_set = load_dataset(name)(train_options)
wild_sampler = data.distributed.DistributedSampler(train_set)
wild_dataloader = data.DataLoader(train_set, shuffle=False,
                                            sampler=wild_sampler,
                                            num_workers=data_cfg.get("num_workers", 4),
                                            batch_size=16)

val_branch = data_cfg["data"]["val_branch"]
val_options = options[val_branch]
val_set = load_dataset(name)(val_options)
wild_val_sampler = data.distributed.DistributedSampler(val_set)
wild_val_dataloader = data.DataLoader(val_set, shuffle=False,
                                            sampler=wild_val_sampler,
                                            num_workers=data_cfg.get("num_workers", 4),
                                            batch_size=16)

train_dataset =all_data[2]
branch =data_cfg["data"]["train_branch"]
name =all_name[2]
with open(train_dataset, "r") as f:
    options = yaml.load(f, Loader=yaml.FullLoader)
val_branch = data_cfg["data"]["val_branch"]
val_options = options[val_branch]
val_set = load_dataset(name)(val_options)
dfdc_val_sampler = data.distributed.DistributedSampler(val_set)
dfdc_val_dataloader = data.DataLoader(val_set, shuffle=False,
                                            sampler=dfdc_val_sampler,
                                            num_workers=data_cfg.get("num_workers", 4),
                                            batch_size=16)

# train_dataset =all_data[2]
# branch =data_cfg["data"]["train_branch"]
# name =all_name[2]
# with open(train_dataset, "r") as f:
#     options = yaml.load(f, Loader=yaml.FullLoader)
# train_options = options[branch]
# train_set = load_dataset(name)(train_options)
# wild_sampler = data.distributed.DistributedSampler(train_set)
# wild_dataloader = data.DataLoader(train_set, shuffle=False,
#                                             sampler=wild_sampler,
#                                             num_workers=data_cfg.get("num_workers", 4),
#                                             batch_size=256)
#all_data = [dataloader_df, dataloader_ff]

from torch.utils.data import ConcatDataset, DataLoader

# # 创建 CelebDF 和 FaceForensics 数据集实例
# dataset_df = CelebDF(config_df)
# dataset_FF = FaceForensics(config_f)

# 合并数据集
combined_dataset = ConcatDataset([train_set, val_set])

# 创建 DataLoader 以加载合并后的数据集
combined_dataloader = DataLoader(combined_dataset, batch_size=16)

# 使用 combined_dataloader 迭代
for batch in combined_dataloader:
    combined_data, combined_labels = batch