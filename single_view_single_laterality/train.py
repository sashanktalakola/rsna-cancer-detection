import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import torch.nn as nn
from model import Model
from dataset import getDataloader, getClassWeights
import torch.optim as optim
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", choices=['kaggle', 'local'])
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("-l", "--laterality", type=str, default="L")
parser.add_argument("-v", "--view", type=str, default="MLO")
parser.add_argument("-ih", "--height", type=int, default=768)
parser.add_argument("-w", "--width", type=int, default=384)
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-bb", "--backbone", default="seresnext50_32x4d")
parser.add_argument("-p", "--pretrained", action="store_true")
parser.add_argument("-n", "--epochs", type=int, default=10)
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
args = parser.parse_args()

DF_PATH = None
IMG_DIR = None

IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH = args.height, args.width

if args.env == "local":
    DF_PATH = "./train.csv"
    IMG_DIR = "./data-cut-off"
else:
    DF_PATH = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
    IMG_DIR = "/kaggle/input/rsna-cut-off-empty-space-from-images"

df = pd.read_csv(DF_PATH)
df.drop(["BIRADS", "density"], axis=1, inplace=True)

if args.debug:
    df = df.sample(frac=0.05).reset_index(drop=True)

imputer = SimpleImputer(strategy='median')
df.age = imputer.fit_transform(df.age.values.reshape(-1, 1))

LATERALITY = args.laterality
VIEW = args.view
df = df[(df.view == VIEW) & (df.laterality == LATERALITY)]

train_df, valid_df = train_test_split(df, test_size=.2, random_state=42)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

BATCH_SIZE = args.batch_size

train_dataloader = getDataloader(train_df, IMG_DIR, BATCH_SIZE, "TRAIN", "TRAIN", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH)
valid_dataloader = getDataloader(valid_df, IMG_DIR, BATCH_SIZE, "TRAIN", "VALID", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH)

BACKBONE = args.backbone
FEATURE_VEC_SIZE = getFeatureVectorSize(BACKBONE)

model = Model(BACKBONE, FEATURE_VEC_SIZE, pretrained=args.pretrained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

N_ACCUMULATION_STEPS = max(1, 64//BATCH_SIZE)
scaler = torch.cuda.amp.GradScaler()
CSV_LOG_FILE = "{}_{}_{}.csv".format(BACKBONE, LATERALITY, VIEW)
createCSV(CSV_LOG_FILE)

LR = args.learning_rate
LR_PATIENCE = 1
LR_FACTOR = 0.3333
class_weights = getClassWeights(train_df)
weight = torch.Tensor([float(class_weights[1]), ]).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_PATIENCE,
                                                 verbose=True, factor=LR_FACTOR)
scaler = torch.cuda.amp.GradScaler()

print("\nTraining Begin\n")
NUM_EPOCHS = args.epochs
for epoch in range(NUM_EPOCHS):
    train_vals = train(model, train_dataloader, loss_fn, optimizer, epoch, NUM_EPOCHS, N_ACCUMULATION_STEPS, device, scaler)
    valid_vals = valid(model, valid_dataloader, loss_fn, scheduler, epoch, NUM_EPOCHS, device)

    vals = [epoch, ] + train_vals + valid_vals
    writeCSVLog(vals, CSV_LOG_FILE)
    model_path = "{}_{}_{}_{}.pth".format(LATERALITY, VIEW, BACKBONE, epoch+1)
    saveModel(model, optimizer, model_path)
