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


def get_loaders(dataset, batch_size, train_ids, validation_ids):

    trainset = torch.utils.data.Subset(dataset, train_ids)
    validationset = torch.utils.data.Subset(dataset, validation_ids)

    print("Trainset:", len(trainset))
    print("Validationset:",len(validationset))
    
    # Balance trainset #
    class_weights = [1,1.9]
    trainsample_weights = [0]*len(trainset)
    for i, sample in enumerate(trainset):
        class_weight = class_weights[sample['labels']]
        trainsample_weights[i] = class_weight

    trainsampler = WeightedRandomSampler(trainsample_weights, num_samples = len(trainsample_weights), replacement=True)

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=trainsampler)
    validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True)

    return trainloader, validationloader
 

def train_model(device, config):
    
    for e in config['experiments']:

        dataset = PatientDataset('D:/hypertrophy/data/train80/', transform=PatientDataset.augmentation)

        kfold = KFold(n_splits=e['k_folds'], shuffle=True)

        for fold, (train_ids, validation_ids) in enumerate(kfold.split(dataset)):

            trainloader, validationloader =  get_loaders(dataset=dataset, batch_size=e['batch_size'], train_ids=train_ids, validation_ids=validation_ids)

            num_epochs = e['num_epochs']

            model = Conv5Lin3withSigmoid()
            model.to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=e['lr'])

            writer, path_to_save_model = define_writer_and_pathToSaveModel(model, bs=e['batch_size'], ne=e['num_epochs'], lr=e['lr'])

            best_model_state = 0
            best_loss_state = 0.5
            
            step = 0
            
            for epoch in range(num_epochs):

                ###################
                # Train the model #
                ###################
                model.train()
                for i, batch in enumerate(trainloader):
                    images = batch['images'].to(device)
                    labels = batch['labels'].to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    
                    outputs = torch.squeeze(outputs)
                    labels = labels.to(torch.float32)
                    
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                ######################
                # Validate the model #
                ###################### 
                model.eval()
                with torch.no_grad():

                    running_trainLoss = 0
                    running_trainAcc = 0
                    running_trainPrec = 0
                    running_trainRec = 0
                    running_trainF1 = 0

                    running_validationLoss = 0
                    running_validationAcc = 0
                    running_validationPrec = 0
                    running_validationRec = 0
                    running_validationF1 = 0

                    for i, batch in enumerate(trainloader):
                        images = batch['images'].to(device)
                        labels = batch['labels'].to(device)

                        optimizer.zero_grad()
                        outputs = model(images)
                        outputs = torch.squeeze(outputs)
                        labels = labels.to(torch.float32)
                        loss = criterion(outputs, labels)

                        predicted = (outputs > 0.5).to(torch.float32)

                        acc, prec, rec, f1 = get_metrics(labels=labels, predicted=predicted)

                        running_trainLoss += loss.item()
                        running_trainAcc += acc
                        running_trainPrec += prec
                        running_trainRec += rec
                        running_trainF1 += f1


                    for i, batch in enumerate(validationloader):
                        images = batch['images'].to(device)
                        labels = batch['labels'].to(device)

                        optimizer.zero_grad()
                        outputs = model(images)
                        outputs = torch.squeeze(outputs)
                        labels = labels.to(torch.float32)
                        loss = criterion(outputs, labels)

                        predicted = (outputs > 0.5).to(torch.float32)

                        acc, prec, rec, f1 = get_metrics(labels=labels, predicted=predicted)

                        running_validationLoss += loss.item()
                        running_validationAcc += acc
                        running_validationPrec += prec
                        running_validationRec += rec
                        running_validationF1 += f1

                    writer.add_scalars('Loss', {'Train': running_trainLoss/len(trainloader), 'Validation': running_validationLoss/len(validationloader)}, global_step=step)
                    writer.add_scalars('Accuracy', {'Train': running_trainAcc/len(trainloader), 'Validation': running_validationAcc/len(validationloader)}, global_step=step)
                    writer.add_scalars('Precision', {'Train': running_trainPrec/len(trainloader), 'Validation': running_validationPrec/len(validationloader)}, global_step=step)
                    writer.add_scalars('Recall', {'Train': running_trainRec/len(trainloader), 'Validation': running_validationRec/len(validationloader)}, global_step=step)
                    writer.add_scalars('F1-score', {'Train': running_trainF1/len(trainloader), 'Validation': running_validationF1/len(validationloader)}, global_step=step)
                    
                    print("[Fold: {}/{} | Epoch: {}/{}] \nTrain ------> Acc: {:.2f}% | Loss: {:.4f} | Prec: {:.2f}% | Rec: {:.2f}% | F1: {:.2f}% \nValidation -> Acc: {:.2f}% | Loss: {:.4f} | Prec: {:.2f}% | Rec: {:.2f}% | F1: {:.2f}% "
                        .format(fold+1, e['k_folds'], epoch+1, num_epochs, 
                            100*running_trainAcc/len(trainloader),  
                            running_trainLoss/len(trainloader), 
                            100*running_trainPrec/len(trainloader),
                            100*running_trainRec/len(trainloader),
                            100*running_trainF1/len(trainloader),
                            100*running_validationAcc/len(validationloader), 
                            running_validationLoss/len(validationloader),
                            100*running_validationPrec/len(validationloader),
                            100*running_validationRec/len(validationloader),
                            100*running_validationF1/len(validationloader)))
                    step+=1

                
                if(running_validationLoss/len(validationloader) < best_loss_state):
                    best_loss_state = running_validationLoss/len(validationloader)
                    torch.save(model.state_dict(), "models/" + path_to_save_model + ".pt")
                    print("Model saved!")



def get_config(filename):
    with open (filename) as f:
        config = json.load(f)
        return config

def define_writer_and_pathToSaveModel(model, bs, ne, lr):
    now = datetime.now()
    path =  model.__class__.__name__+'_bs'+str(bs)+'_ne'+str(ne)+'_lr'+str(lr).split(".")[1]+'_'+now.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/"+path)
    return writer, path


def get_metrics(labels, predicted):
    # Move tensors back to cpu for sklearn.metrics #
    labels = labels.cpu()
    predicted = predicted.cpu()

    acc = sklearn.metrics.accuracy_score(labels, predicted)
    prec = sklearn.metrics.precision_score(labels, predicted, pos_label=1, zero_division=0)
    rec = sklearn.metrics.recall_score(labels, predicted, pos_label=1, zero_division=0)
    f1 = sklearn.metrics.f1_score(labels, predicted, pos_label=1, zero_division=0)

    return acc, prec, rec, f1


def main():
    device = torch.device('cuda')
    #device=torch.device('cpu')
    print("Device:", device)
    config = get_config('trainingConfigurations.json')
    train_model(device=device, config=config)

if __name__ == "__main__":
    main()