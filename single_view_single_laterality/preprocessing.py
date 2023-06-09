import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Resize, Normalize, HorizontalFlip, Rotate, RandomResizedCrop

def getTransforms(mode="TRAIN", IMG_SIZE_HEIGHT=768, IMG_SIZE_WIDTH=384):
    if mode == "TRAIN":
        transforms = A.Compose([
            HorizontalFlip(p=0.5),
            Rotate(limit=5, always_apply=True),
            RandomResizedCrop(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, scale=(0.8, 1), ratio=(0.45, 0.55)),
            Normalize(mean=0.2179, std=0.0529),
            ToTensorV2(),
            ])
    else:
        transforms = A.Compose([
            Resize(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH),
            Normalize(mean=0.2179, std=0.0529),
            ToTensorV2(),
        ])
    return transforms