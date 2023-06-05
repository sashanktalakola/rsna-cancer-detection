import csv
import torch
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score


def createCSV(csv_file_path):
    csv_writer = None
    cols = ["epoch", "training_loss", "training_acc", "training_f1", "validation_loss",
            "validation_acc", "validation_f1"]
    with open(csv_file_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(cols)

def writeCSVLog(vals, csv_file_path):
    with open(csv_file_path, "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(vals)

def train(model, dataloader, loss_fn, optimizer, epoch, num_epochs, N_ACCUMULATION_STEPS, device, scaler):
    model.train()
    losses = []
    mean_accuracy = 0.
    mean_f1_score = 0.
    accuracy = Accuracy(task="binary").to(device)
    f1_score = F1Score(task="binary").to(device)

    loop = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_idx, (X, y) in loop:
        loop.set_description("Epoch [{}/{}]".format(epoch+1, num_epochs))

        # model = model.to(device)
        X = X.to(device)
        y = y.to(device)

        with torch.cuda.amp.autocast():
            logits = model(X)
            loss = loss_fn(logits, y.unsqueeze(1))
        losses.append(loss.item())
        loss = loss / N_ACCUMULATION_STEPS
        scaler.scale(loss).backward()

        if ((batch_idx + 1) % N_ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        probabilities = torch.sigmoid(logits).squeeze()
        accuracy_value = accuracy(probabilities, y).item()
        mean_accuracy += accuracy_value
        f1_score_value = f1_score(probabilities, y).item()
        mean_f1_score += f1_score_value

        loop.set_postfix(loss=loss.item(), acc=accuracy_value, f1=f1_score_value)

    mean_loss = sum(losses) / len(losses)
    mean_accuracy /= len(losses)
    mean_f1_score /= len(losses)

    return [mean_loss, mean_accuracy, mean_f1_score]


def valid(model, dataloader, loss_fn, scheduler, epoch, num_epochs, device):
    model.eval()
    losses = []
    mean_accuracy = 0.
    mean_f1_score = 0.
    accuracy = Accuracy(task="binary").to(device)
    f1_score = F1Score(task="binary").to(device)

    with torch.inference_mode():
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (X, y) in loop:
            loop.set_description("Epoch [{}/{}]".format(epoch+1, num_epochs))

            # model = model.to(device)
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y.unsqueeze(1))
            losses.append(loss.item())

            probabilities = torch.sigmoid(logits).squeeze()
            accuracy_value = accuracy(probabilities, y).item()
            mean_accuracy += accuracy_value
            f1_score_value = f1_score(probabilities, y).item()
            mean_f1_score += f1_score_value

            loop.set_postfix(loss=loss.item(), acc=accuracy_value, f1=f1_score_value)

        mean_loss = sum(losses) / len(losses)
        mean_accuracy /= len(losses)
        mean_f1_score /= len(losses)
        scheduler.step(mean_loss)

    return [mean_loss, mean_accuracy, mean_f1_score]

def saveModel(model, optimizer,  path):
    state_dict = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict()}
    torch.save(state_dict, path)
