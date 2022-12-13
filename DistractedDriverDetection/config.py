#Specify all the hyperparameters of the model

import torch
#import albumentations as A
#from albumentations.pytorch import ToTensorV2



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 20
NUM_EPOCHS = 20 #100
NUM_WORKERS = 6
CHECKPOINT_FILE = "mobilenetv2.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

# The goal here is create the most simple baseline (we can expand lately)
# So we do simple transfromes not Horizontal, Vertical thing

