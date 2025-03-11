import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from byol import BYOL
import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from pytorch_lightning.callbacks import ModelCheckpoint


resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 2) 
weights_path = "./resnet50_tumor_classifier.pth" 
state_dict = torch.load(weights_path,)
# 加载权重到模型
resnet.load_state_dict(state_dict)

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, 
                    default="",
                    help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants

BATCH_SIZE = 16
EPOCHS     = 200
LR         = 1e-4
NUM_GPUS   = 1
IMAGE_SIZE = 512
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
# NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = 1

# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log('loss', loss)  
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for label in [0, 1]:  
            for path in Path(f'{folder}/{label}').glob('*'): 
                _, ext = os.path.splitext(path)
                if ext.lower() in IMAGE_EXTS:
                    self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([

            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        if img.mode == 'L':
            img = img.convert('RGB')  
        return self.transform(img)

class LossCallback(Callback):
    def __init__(self):
        self.epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics["loss"].item()
        self.epoch_losses.append(loss)

# main

if __name__ == '__main__':
    ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='loss',  
        dirpath='./checkpoints/', 
        filename='{epoch}-{loss:.2f}', 
        save_top_k=1,  
        mode='min',  
        save_last=True, 
        every_n_epochs=1 
    )

    loss_callback = LossCallback()
   
    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = True,
        callbacks=[checkpoint_callback, loss_callback],  
        log_every_n_steps=40,
    )

    trainer.fit(model, train_loader)

    best_model_path = checkpoint_callback.best_model_path