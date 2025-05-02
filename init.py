from abstract_dataset import AbstractDataset
from faceforensics import FaceForensics
from wild_deepfake import WildDeepfake
from celeb_df import CelebDF
from dfdc import DFDC
from dff import DFF
LOADERS = {
    "FaceForensics": FaceForensics,
    "WildDeepfake": WildDeepfake,
    "CelebDF": CelebDF,
    "DFDC": DFDC,
    "DFF": DFF,
}


def load_dataset(name="FaceForensics"):
    print(f"Loading dataset: '{name}'...")
    return LOADERS[name]