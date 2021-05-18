from sklearn.model_selection import KFold
import torch
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, utils
from Conv5Lin3withSigmoid import Conv5Lin3withSigmoid
import sklearn.metrics
import decimal
from PatientDataset import PatientDataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchmetrics

def main():
    device = torch.device('cuda')
    #device=torch.device('cpu')
    print("Device:", device)

    model = Conv5Lin3withSigmoid()
    criterion = nn.BCELoss()
    model.to(device)
    #model.load_state_dict(torch.load("models/Conv5Lin3withSigmoid_bs64_ne100_lr0001_20210514_191851.pt"))
    model.load_state_dict(torch.load("models/Conv5Lin3withSigmoid_bs64_ne100_lr0005_20210428_105217.pt"))
    model.eval()

    #testset = PatientDataset('D:/hypertrophy/data/datasetsSTATIC/testset/')
    testset = PatientDataset('D:/hypertrophy/data/test20/')
    testloader = DataLoader(testset, batch_size=600, shuffle=True)

    running_testLoss = 0
    running_testAcc = 0
    running_testPrec = 0
    running_testRec = 0
    running_testF1 = 0

    with torch.no_grad():
        for i, batch in enumerate(testloader):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            #optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            labels = labels.to(torch.float32)
            loss = criterion(outputs, labels)

            predicted = (outputs > 0.5).to(torch.float32)

            acc, prec, rec, f1 = get_metrics(labels=labels, predicted=predicted)

            running_testLoss += loss.item()
            running_testAcc += acc
            running_testPrec += prec
            running_testRec += rec
            running_testF1 += f1

        print("Loss: {:.4f}  Accuracy: {:.2f}  Precision: {:.2f}  Recall: {:.2f}  F1-score: {:.2f} "
            .format(
                running_testLoss/len(testloader),
                100*running_testAcc/len(testloader),
                100*running_testPrec/len(testloader),
                100*running_testRec/len(testloader),
                100*running_testF1/len(testloader)))

def get_metrics(labels, predicted):
    # Move tensors back to cpu for sklearn.metrics #
    labels = labels.cpu()
    predicted = predicted.cpu()

    acc = sklearn.metrics.accuracy_score(labels, predicted)
    prec = sklearn.metrics.precision_score(labels, predicted, pos_label=1, zero_division=0)
    rec = sklearn.metrics.recall_score(labels, predicted, pos_label=1, zero_division=0)
    f1 = sklearn.metrics.f1_score(labels, predicted, pos_label=1, zero_division=0)

    return acc, prec, rec, f1

if __name__ == "__main__":
    main()