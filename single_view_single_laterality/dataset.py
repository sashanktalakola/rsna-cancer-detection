import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import compute_class_weight
from preprocessing import *

class RSNADataset(Dataset):
    def __init__(self, df, img_dir, mode="TRAIN", transforms_mode="TRAIN", IMG_SIZE_HEIGHT=768, IMG_SIZE_WIDTH=384):
        self.df = df
        self.img_dir = img_dir
        self.mode = mode
        self.transforms_mode = transforms_mode
        self.transforms = getTransforms(transforms_mode, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]
        image_id = row.image_id
        patient_id = row.patient_id
        y_label = row.cancer.astype(np.float32)

        image_name = "{}/{}.png".format(row.patient_id, row.image_id)
        image_path = os.path.join(self.img_dir, image_name)
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        img = self.transforms(image=img)["image"]

        if self.mode == "TRAIN":
            return torch.tensor(img, dtype=torch.float), torch.tensor(y_label, dtype=torch.float)
        else:
            return torch.tensor(img, dtype=torch.float)

def getClassWeights(df):
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(df.cancer),
                                         y=df.cancer)
    class_weights = dict(zip(np.unique(df.cancer), class_weights))
    return class_weights

def getDataloader(df, img_dir, batch_size, mode="TRAIN", transforms_mode="TRAIN", IMG_SIZE_HEIGHT=768, IMG_SIZE_WIDTH=384):
    dataset = RSNADataset(df, img_dir, mode, transforms_mode, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH)
    if transforms_mode == "TRAIN":
        class_weights = getClassWeights(df)
        sample_weights = [0] * len(dataset)
        loop = tqdm(enumerate(dataset), total=len(dataset))
        loop.set_description("Random Sampler")
        for idx, (_, y) in loop:
            class_weight = class_weights[y.item()]
            sample_weights[idx] = class_weight
            if idx == len(dataset)-1: break
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size)