import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Resize

def getTransforms(mode="TRAIN", IMG_SIZE_HEIGHT=768, IMG_SIZE_WIDTH=384):
    if mode == "TRAIN":
        transforms = A.Compose([
                Resize(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH),
                ToTensorV2()
            ])
    else:
        transforms = A.Compose([
            Resize(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH),
            ToTensorV2(),
        ])
    return transforms