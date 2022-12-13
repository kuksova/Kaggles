# loading model
# creating submissoin file as well
# create prediction function
# scoring function. What metric to use?

import torch
import pandas as pd
import numpy as np
import config
from tqdm import tqdm


def make_prediction(model, loader):
    probs = []
    filenames = []
    model.eval()

    for x, files in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            scores = model(x) # use cross entropy loss as simple
            probs1 = torch.softmax(scores, dim=0)
            #nm = probs.sum(dim=0) # is equal 1
            probs.append((probs1.cpu().numpy()))
            filenames +=files

    # Submission file
    predictions_out = pd.DataFrame(columns=['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    for i in range(len(probs)):
        predictions_out.loc[i, 'img'] = filenames[i]

        predictions_out.loc[i, 'c0':'c9'] = probs[i]

    predictions_out.to_csv('submission.csv', index=False)

    #df = pd.DataFrame() # how to this through column?
    # df = pd.DataFrame(list(zip(*[list1, list2, list3]))).add_prefix('Col')
    #df = pd.DataFrame({"img": filenames, "likelihood": probs}).add_prefix('c')
    print("Done with predictions")

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0

    for i_step, (x, y) in enumerate(loader): #x,y in tqdm(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct +=(predictions == y).sum()
            num_samples += predictions.shape[0]

            # add to lists to be going to send to sklearn to
            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

        #model.train()
        kappa = np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(
            all_labels, axis=0, dtype=np.int64)
        accur = float(num_correct) / num_samples
        return kappa, accur

""""
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
"""
