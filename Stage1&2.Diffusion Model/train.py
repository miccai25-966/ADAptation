from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from tutorial_dataset import MyDataset
import torch
import torch.nn as nn

batch_size = 4
logger_freq = 30
learning_rate = 1e-4
sd_locked = True
only_mid_control = False
max_epochs = 200
acc_grad = 4
model = create_model('./models/cldm_v15.yaml').cpu()

#### load the SD inpainting weights
states = load_state_dict("./models/v1-5-pruned.ckpt", location='cpu')
model.load_state_dict(states, strict=False)

control_states = load_state_dict("./models/control_sd15_canny.pth", location='cpu')
model.load_state_dict(control_states, strict=False)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset("./train.json")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

trainer = pl.Trainer(precision=32, max_epochs=max_epochs, accelerator="gpu", gpus=2, callbacks=[logger], accumulate_grad_batches=acc_grad, strategy='ddp')

# Train!
trainer.fit(model, dataloader)
