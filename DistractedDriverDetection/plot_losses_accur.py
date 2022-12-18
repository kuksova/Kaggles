# Summary of Loss & Accuracy

import matplotlib.pyplot as plt

def plot_losses(val_losses, train_losses):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation LOSS for this model")
    plt.plot(val_losses,label="val")
    plt.plot(train_losses,label="train")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def plot_accuracy(val_accur, train_accur):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation ACCURACY for this model")
    plt.plot(val_accur,label="val")
    plt.plot(train_accur,label="train")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()