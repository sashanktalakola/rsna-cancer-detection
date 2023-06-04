import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Resize

IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH = 768, 384
def getTransforms(mode="TRAIN"):
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