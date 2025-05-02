import cv2
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import pdb
# text_mapping = {
#     0: "A face that exists in reality",
#     1: "A face created by AI"
# }
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# text_mapping = {
#     0: "a photo of real face",
#     1: "a photo of fake face"
# }
text_mapping = {
    0: "This is a real person",
    1: "This is a fake person"
}

text_mapping2 = {
    0: "real",
    1: "fake"
}
import clip
from PIL import Image

class AbstractDataset(VisionDataset):
    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        super(AbstractDataset, self).__init__(cfg['root'], transforms=transforms,
                                              transform=transform, target_transform=target_transform)
        # fix for re-production
        np.random.seed(2022)

        self.images = list()
        self.targets = list()
        self.split = cfg['split']
        if self.transforms is None:
            self.transforms = Compose(
                [getattr(albumentations, _['name'])(**_['params']) for _ in cfg['transforms']] +
                [ToTensorV2()]
            )
        self.preprocess=preprocess
        #self.title = clip.tokenize(list_txt)#self.title[idx]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #print(self.images[index])
        path = self.images[index]
        types=None
        if("Deepfakes" in path):
            types=1
        elif("Face2Face" in path):
            types=2
        elif("FaceSwap" in path):
            types=3
        elif("NeuralTextures" in path):
            types=4
        else:
            types=0
        image = self.preprocess(Image.open(path))  # Image from PIL module
        tgt = self.targets[index]
        text = text_mapping[tgt]
        text2=text_mapping2[tgt]
        text=clip.tokenize(text)
        text2=clip.tokenize(text2)
        return image, tgt, text,text2

    def load_item(self, items):
        images = list()
        for item in items:
            img = cv2.imread(item)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=img)['image']
            images.append(image)
        return torch.stack(images, dim=0)

    def get_text(self):
        mapping_text = []
        for tgt in range(len(text_mapping)):
            text = text_mapping[tgt]
            text=clip.tokenize(text)
            mapping_text.append(text)
        return torch.cat(mapping_text, dim=0)
