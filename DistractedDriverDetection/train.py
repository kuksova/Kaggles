# to train our model

#from sklearn.metrics import cohen_kappa_score


import torch
from torch import nn, optim
#import os
import config
#from torch.utils.data import DataLoader
#from tqdm import tqdm # iterator and progress bar
from sklearn.metrics import cohen_kappa_score

from dataset import load_datasets
#from torchvision.utils import save_image
from utils import (
    check_accuracy,
    make_prediction # for submission file
)
from plot_losses_accur import plot_losses, plot_accuracy
import torchvision.models as models
#from efficient_pytorch import EfficientNet



def train_one_epoch(trainloader, model, optimizer, criterion, device):
    losses = []
    loss_accum = 0
    correct_samples = 0
    total_samples = 0
    for batch_idx, data in enumerate(trainloader):
        inputs, labels = data # data is a list of [inputs, labels]

        # both the model and the data must exist on the same device
        inputs = inputs.to(device=device) # GPU 3
        labels = labels.to(device=device) #GPU 4


        # forward + optimize
        outputs = model(inputs)
        loss_value = criterion(outputs, labels)
        optimizer.zero_grad()  # zero the parameter gradients
        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())
        loss_accum += loss_value

        # accuracy
        _, indices = torch.max(outputs, 1)
        correct_samples += torch.sum(indices == labels)
        total_samples += labels.shape[0]
        train_accuracy = float(correct_samples) / total_samples

        # backward

        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        #loop.set_postfix(loss=loss.item())

        # print statistics
        #running_loss += loss.item()
        #if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
        #    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #    running_loss = 0.0
    #ave_loss = loss_accum/batch_idx
    #print("batch_idx ", batch_idx)
    #print('len(losses) ', len(losses))
    ave_loss = sum(losses) / len(losses)
    #print(f"Loss average over epoch: {sum(losses) / len(losses)}")
    return ave_loss, train_accuracy

def main():
    # Load datasets, create data_loaders
    # Create loss functions, model,

    #All loaders
    train_root = '/home/sveta/PycharmProjects/Kaggles/ComputerVision/StateFarm/state_farm_img/train'
    test_root = '/home/sveta/PycharmProjects/Kaggles/ComputerVision/StateFarm/state_farm_img/test'
    train_dataset_loader, val_loader, test_loader = load_datasets(train_root, test_root)

    #model = EfficientNet.from_pretrained("efficinet-b3")
    model = models.resnet50(pretrained=True) #models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    class_names = 10 #len(train_ds)
    model.fc = nn.Linear(num_ftrs, class_names)

    model = model.to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    #optimizer = optim.SGD(model.parameters(), lr=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    #scaler

    loss_train_history = []
    loss_val_history = []
    accur_train_history = []
    accur_val_history = []
    for epoch in range(config.NUM_EPOCHS):
        print("Run epoch ", epoch)
        ave_loss_per_epoch, train_accuracy = train_one_epoch(train_dataset_loader,model, optimizer,loss_fn,config.DEVICE)
        loss_train_history.append(ave_loss_per_epoch)
        accur_train_history.append((train_accuracy))

        # get on validation
        ab, val_accuracy, val_loss  = check_accuracy(val_loader, model, loss_fn, config.DEVICE)
        accur_val_history.append(val_accuracy)
        loss_val_history.append(val_loss)

        preds, labels = ab
        #print(f"QuadraticWeightedKappa (Validation): {cohen_kappa_score(labels, preds, weights='quadratic')}")
        val_accuracy1 = cohen_kappa_score(labels, preds, weights='quadratic')

        # get on train
        #preds, labels = check_accuracy(train_dataset_loader, model, config.DEVICE)
        #train_accuracy = cohen_kappa_score(labels, preds, weights='quadratic')
        #print(f"QuadraticWeightedKappa (Training): {cohen_kappa_score(labels, preds, weights='quadratic')}")
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss_per_epoch, train_accuracy, val_accuracy))


    #make_prediction(model, test_loader) # output_csv="submission.csv"
    plot_losses(loss_val_history, loss_train_history) #val, train
    plot_accuracy(accur_val_history, accur_train_history)

if __name__ == "__main__":
    main()
