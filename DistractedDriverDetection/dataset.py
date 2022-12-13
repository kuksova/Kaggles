# file to laod the data
import config
import os
#import pandas as pd
import numpy as np

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
#from tqdm import tqdm


train_transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], # All Pytorch pre-trained models expect input images normalized in the same way,
                    std=[0.229, 0.224, 0.225])

    ])
# Do I need custom dataset? Or I can use generic datasret

driver_dataset = datasets.ImageFolder(root='./state_farm_img/train',
                                           transform=train_transform)

data_size = len(driver_dataset)
validation_split = .2
split = int(np.floor(validation_split * data_size))
indices = list(range(data_size))
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_dataset_loader = torch.utils.data.DataLoader(driver_dataset,
                                             batch_size=config.BATCH_SIZE, shuffle=False, sampler=train_sampler) # num_workers=config.NUM_WORKERS

val_loader = torch.utils.data.DataLoader(driver_dataset, batch_size=config.BATCH_SIZE,
                                         sampler=val_sampler)


class TestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_file = self.image_files[idx]
        image = Image.open(os.path.join(self.root_dir, image_file))

        if self.transform:
            image = self.transform(image)

        return image, image_file


test_dataset = TestDataset(root_dir='./state_farm_img/test',
                                           transform=train_transform)
test_loader = torch.utils.data.DataLoader(test_dataset) # batch_size=config.BATCH_SIZE

#class DriversDataset(Dataset)
#for x, label in tqdm(dataset_loader):
#    print(x.shape)
#    print(label.shape)

#for i_batch, sample_batched in enumerate(train_dataset_loader):
#    print(i_batch, sample_batched[0].shape)
