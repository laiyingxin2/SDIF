import cv2
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from skimage import io
import os
from PIL import Image

class AbstractDataset(VisionDataset):
    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        super(AbstractDataset, self).__init__(cfg['root'], transforms=transforms,
                                              transform=transform, target_transform=target_transform)
        # fix for re-production
        np.random.seed(seed)

        self.images = list()
        self.targets = list()
        self.split = cfg['split']
        if self.transforms is None:
            self.transforms = Compose(
                [getattr(albumentations, _['name'])(**_['params']) for _ in cfg['transforms']] +
                [ToTensorV2()]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
#         import pdb
#         pdb.set_trace()
        path = self.images[index]
        #print(path)
        #img = cv2.imread(path)
        if(".ipynb_checkpoints" in path):
            path="pickle_files/wiki/01/2201_1922-06-19_1955.jpg"
            # import pdb
            # pdb.set_trace()
        #img=io.imread(path).astype(np.float32)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=Image.open(path)
        # if(img) is None:
        #     img=Image.open("/home/data/lyx/RECCE/Detect_and_Locate/pickle_files/wiki/01/2201_1922-06-19_1955.jpg")
        img = img.convert("RGB")
        img=np.array(img)
        image = self.transforms(image=img)['image']
        tgt = self.targets[index]
#         print(path)
#         print(tgt)
#         import pdb
#         pdb.set_trace()
        return image, tgt

    # def load_item(self, items):
    #     images = list()
    #     for item in items:
    #         img = cv2.imread(item)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         image = self.transforms(image=img)['image']
    #         images.append(image)
    #     return torch.stack(images, dim=0)
