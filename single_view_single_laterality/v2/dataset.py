import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import compute_class_weight
from preprocessing import *


class RSNADatasetAux(Dataset):
    def __init__(self, df, img_dir, aux_features, mode="TRAIN", transforms_mode="TRAIN", IMG_SIZE_HEIGHT=768, IMG_SIZE_WIDTH=384):
        self.df = df
        self.img_dir = img_dir
        self.aux_features = aux_features
        self.mode = mode
        self.transforms_mode = transforms_mode
        self.transforms = getTransforms(transforms_mode, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]
        y_label = row.cancer.astype(np.float32)
        y_auxiliary = self.df.loc[i, self.aux_features]

        image_name = "{}/{}.png".format(row.patient_id, row.image_id)
        image_path = os.path.join(self.img_dir, image_name)
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        img = self.transforms(image=img)["image"]
        img = torch.tensor(img, dtype=torch.float)
        img = torch.permute(img, (-1, 0, 1))

        if self.mode == "TRAIN":
            return img, torch.tensor(y_label, dtype=torch.float), torch.tensor(y_auxiliary, dtype=torch.float)
        else:
            return img

def getClassWeights(df):
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(df.cancer),
                                         y=df.cancer)
    class_weights = dict(zip(np.unique(df.cancer), class_weights))
    return class_weights

def getDataloader(df, img_dir, batch_size, aux_features, mode="TRAIN", transforms_mode="TRAIN", IMG_SIZE_HEIGHT=768, IMG_SIZE_WIDTH=384):
    dataset = RSNADatasetAux(df, img_dir, aux_features, mode, transforms_mode, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH)
    if mode == "TRAIN":
        class_weights = getClassWeights(df)
        sample_weights = [0] * len(dataset)
        loop = tqdm(enumerate(dataset), total=len(dataset))
        loop.set_description("Random Sampler")
        for idx, (_, y, _) in loop:
            class_weight = class_weights[y.item()]
            sample_weights[idx] = class_weight
            if idx == len(dataset)-1: break
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size)