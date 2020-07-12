import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import trange
from tqdm import tqdm
from tensorboardX import SummaryWriter
import json
import random
import math


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 152), 
        )
        self.decoder = nn.Sequential(
            nn.Linear(152, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
        )

    def forward(self, x):
        encoded = self.encoder(x) # batch*36 x 152
        decoded = self.decoder(encoded) # batch*36 x 2048
        return encoded, decoded


EPOCH = 10
MINI_BATCH_SIZE = 50
LR = 0.005
num_iter =100

def get_heading_degree():
    new_headings = []
    for i in range(0, 360, 30):
        current_radians = i*math.pi/180
        new_headings.append(current_radians)
    return new_headings


def get_data_batch(inputs, batch_size=None, shuffle=False):
    rows = inputs.shape[0]
    indices = list(range(rows))
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_indices = np.asarray(indices[0:batch_size])
        indices = indices[batch_size:] + indices[:batch_size] 
        yield batch_indices

train_scan_list = []
with open('/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/data/R2R_train.json') as f_train:
    train_scenery = json.load(f_train)
for each_train_scenery in train_scenery:
    if each_train_scenery['scan'] not in train_scan_list:
        train_scan_list.append(each_train_scenery['scan'])
relative_path = "/egr/research-hlr/joslin/Matterdata/v1/scans/pretrained_npy/"
feature = []
autoencoder = AutoEncoder().cuda(device=6)
all_scenery = os.listdir(relative_path)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()
writer = SummaryWriter('runs/experiment_relu')
standard_heading = get_heading_degree()

if __name__ == "__main__":  
# training process
    """
    
    for epoch in trange(EPOCH):
        random.shuffle(train_scan_list)
        for scenry in trange(len(train_scan_list)): # each in 61 scans; scenery is the first index in 5
            temp_feature = []# collect all 1 features
            scenery_nump = np.load(relative_path + train_scan_list[scenry] + ".npy", allow_pickle=True)
            for each_state, value in scenery_nump.item().items():
                for each_heading in standard_heading:    
                    temp_feature.append(torch.tensor(value[each_heading]['features'], device=6))
         
            temp_feature = torch.stack(temp_feature, dim=0)
            temp_feature = temp_feature.view(-1, temp_feature.shape[-1])
            batch = get_data_batch(temp_feature, batch_size=MINI_BATCH_SIZE, shuffle=True)
            for each_iter in range(num_iter): # to get 100 each time in tem_feature     
                batch_indices = next(batch)
                mini_temp_feature =temp_feature[batch_indices]
                encoded, decoded = autoencoder(mini_temp_feature)
                loss = loss_func(decoded, mini_temp_feature)      
                optimizer.zero_grad()             
                loss.backward()                    
                optimizer.step()  
                print("iter" + str(each_iter))
                print('loss' + str(loss))
            writer.add_scalar("Loss/train", loss, epoch*len(train_scan_list)+scenry)
        
        torch.save(autoencoder.state_dict(), "tasks/R2R-pano/model/relu_epoch_" + str(epoch))
    """
# relu device6
    # all_scenery_list = os.listdir(relative_path)
    # all_scenery_list = [i[:-4] for i in all_scenery_list]
    all_scenery_list = ['rqfALeAoiTq']
    model_path = 'tasks/R2R-pano/model/relu_epoch_9'
    autoencoder = AutoEncoder().cuda(device=6)
    autoencoder.load_state_dict(torch.load(model_path))
    all_reduced_dimension = {}
    for scenry in tqdm(all_scenery_list): # each in 61 scans; scenery is the first index in 5
        temp_feature = []# collect all 1 features
        scenery_nump = np.load(relative_path + scenry + ".npy", allow_pickle=True)
        new_scenry = scenery_nump.item()
        for each_state, value in new_scenry.items():
            new_value = {}
            for each_heading in standard_heading:    
                temp_dict = {}
                encoded, decoded = autoencoder(torch.tensor(value[each_heading]['features'],device=6))
                temp_dict['image_h'] = value[float(each_heading)]['image_h']
                temp_dict['image_w'] = value[each_heading]['image_w']
                temp_dict['boxes'] = value[each_heading]['boxes']
                temp_dict['features'] = encoded.detach().cpu().numpy()
                new_value[each_heading] = temp_dict
            new_scenry[each_state] = new_value
        #all_reduced_dimension[scenry] = new_scenry
    
    all_reduced_dimension = np.load("img_features/152-object_feature1.npy", allow_pickle=True).item()
    all_reduced_dimension['rqfALeAoiTq'] = new_scenry
    np.save("img_features/152-object_feature2.npy", all_reduced_dimension)
