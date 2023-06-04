import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
from model import Model
from dataset import getDataloader, getClassWeights
import torch.optim as optim
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", choices=['kaggle', 'local'])
args = parser.parse_args()

DF_PATH = None
IMG_DIR = None

if args.env == "local":
    DF_PATH = "./train.csv"
    IMG_DIR = "./data-cut-off"
else:
    DF_PATH = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
    IMG_DIR = "/kaggle/input/rsna-cut-off-empty-space-from-images"

df = pd.read_csv(DF_PATH)
df.drop(["BIRADS", "density"], axis=1, inplace=True)

#df = df.sample(frac=0.01).reset_index(drop=True)

imputer = SimpleImputer(strategy='median')
df.age = imputer.fit_transform(df.age.values.reshape(-1, 1))

LATERALITY = "L"
VIEW = "MLO"
df = df[(df.view == VIEW) & (df.laterality == LATERALITY)]

train_df, valid_df = train_test_split(df, test_size=.2, random_state=42)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

BATCH_SIZE = 16

train_dataloader = getDataloader(train_df, IMG_DIR, BATCH_SIZE, mode="TRAIN", transforms_mode="TRAIN")
valid_dataloader = getDataloader(valid_df, IMG_DIR, BATCH_SIZE, mode="TRAIN", transforms_mode="VALID")

BACKBONE = "seresnext50_32x4d"
FEATURE_VEC_SIZE = 2048

model = Model(BACKBONE, FEATURE_VEC_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

N_ACCUMULATION_STEPS = max(1, 64//BATCH_SIZE)
scaler = torch.cuda.amp.GradScaler()
CSV_LOG_FILE = "{}_{}_{}.csv".format(BACKBONE, LATERALITY, VIEW)
createCSV(CSV_LOG_FILE)

LR = 3e-5
LR_PATIENCE = 1
LR_FACTOR = 0.4
class_weights = getClassWeights(train_df)
weight = torch.Tensor([float(class_weights[1]), ]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_PATIENCE,
                                                 verbose=True, factor=LR_FACTOR)
scaler = torch.cuda.amp.GradScaler()

print("\nTraining Begin\n")
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    train_vals = train(model, train_dataloader, loss_fn, optimizer, epoch, NUM_EPOCHS, N_ACCUMULATION_STEPS, device, scaler)
    valid_vals = valid(model, valid_dataloader, loss_fn, scheduler, epoch, NUM_EPOCHS, device)

    vals = [epoch, ] + train_vals + valid_vals
    writeCSVLog(vals, CSV_LOG_FILE)
