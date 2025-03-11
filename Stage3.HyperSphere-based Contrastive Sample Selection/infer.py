import torch
from torchvision import models
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
import os
from PIL import Image
import argparse
from torchvision import models, transforms

import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps 
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
    
from byol import BYOLL

# constants
BATCH_SIZE = 1   
NUM_GPUS   = 1 
IMAGE_SIZE = 512
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = 1

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
            
            
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

num_ftrs = resnet.fc.in_features

model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

model_path = "./last.ckpt"
model.eval()  
model.to('cuda')

class TestDataset_real(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []
        for path in Path(f'{folder}').glob('*'): 
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
    
    
class TrainDataset(Dataset):
    def __init__(self, json_path, resolution = 512,):
        self.data = []
        self.resolution = resolution
        
        with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
                image_count = 0
                for item in self.data:
                    source_filename = item.get('source') 
                    if source_filename:
                        if Path(source_filename).exists():
                            image_count += 1
                print(f"{image_count} image paths found in JSON file") 
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):
        item = self.data[index]
        image_path = item["source"]
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB') 
        return self.transform(img)
    
def kmeans_clustering_and_visualization(representation, n_clusters=4):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(representation.cpu().numpy())
    labels = torch.tensor(kmeans.labels_, device=representation.device)
    
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42, n_jobs=-1) # 使用cosine metric更适用于球面数据
    embedding = reducer.fit_transform(representation.cpu().numpy())

    return kmeans, embedding, labels



def main():

    idtraining = ""
    ooddata = ""
    real = TrainDataset(idtraining, IMAGE_SIZE)
    recon = TrainDataset(ooddata, IMAGE_SIZE)

    test_loader_real = DataLoader(real, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    test_loader_recon = DataLoader(recon, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    features_real = []
    features_recon = []
    features_real_sphere = []
    features_recon_sphere = []

    for images in test_loader_real:  
        images = images.to('cuda')
        with torch.no_grad():
            representation_real, _ = model.learner.online_encoder(images)
            features_real.append(representation_real)

            representation_real_sphere = F.normalize(representation_real, dim=-1, p=2)
            features_real_sphere.append(representation_real_sphere)

            concatenated_tensor = torch.cat(features_real_sphere, dim=0)

    final_concat_real = concatenated_tensor 
    final_concat_real.cpu()

    kmeans_model, umap_embedding, labels = kmeans_clustering_and_visualization(final_concat_real, n_clusters=4)

