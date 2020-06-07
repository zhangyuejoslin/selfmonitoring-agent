import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import trange
from tensorboardX import SummaryWriter
import json
import random
import math


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 152), 
        )
        self.decoder = nn.Sequential(
            nn.Linear(152, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
        )

    def forward(self, x):
        encoded = self.encoder(x) # batch*36 x 152
        decoded = self.decoder(encoded) # batch*36 x 2048
        return encoded, decoded


EPOCH = 10
MINI_BATCH_SIZE = 1000
LR = 0.005

train_scan_list = []
with open('/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/data/R2R_train.json') as f_train:
    train_scenery = json.load(f_train)
for each_train_scenery in train_scenery:
    if each_train_scenery['scan'] not in train_scan_list:
        train_scan_list.append(each_train_scenery['scan'])

relative_path = "/egr/research-hlr/joslin/Matterdata/v1/scans/pretrained_npy/"
feature = []
autoencoder = AutoEncoder().cuda(device=1)
all_scenery = os.listdir(relative_path)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()
writer = SummaryWriter('runs/experiment_1')

def get_heading_degree():
    new_headings = []
    for i in range(0, 360, 30):
        current_radians = i*math.pi/180
        new_headings.append(current_radians)
    return new_headings

standard_heading = get_heading_degree()

for epoch in trange(EPOCH):
    random.shuffle(train_scan_list)
    for scenry in trange(0, len(train_scan_list), 1): # each 5 in 61 scans; scenery is the first index in 5
        temp_feature = []# collect all 5 features
        for each_scenery in train_scan_list[scenry:min(scenry+4, len(train_scan_list))]: # get feature from each scenery in 5 sceneries
            scenery_nump = np.load(relative_path + each_scenery + ".npy", allow_pickle=True)
            for each_state, value in scenery_nump.item().items():
                for each_heading in standard_heading:    
                    temp_feature.append(torch.tensor(value[each_heading]['features'], device=1))
        #for mini_batch in range(MINI_BATCH_SIZE):
        temp_feature = torch.stack(temp_feature, dim=0)
        temp_feature = temp_feature.view(-1, temp_feature.shape[-1])
        encoded, decoded = autoencoder(temp_feature)
        loss = loss_func(decoded, temp_feature)      
        optimizer.zero_grad()             
        loss.backward()                    
        optimizer.step()  

        print(loss)
        writer.add_scalar("Loss/train", loss, epoch)


torch.save(autoencoder.state_dict(), "tasks/R2R-pano/run")

