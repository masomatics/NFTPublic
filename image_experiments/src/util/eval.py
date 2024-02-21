import math
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import os
import yaml

__LOG10 = math.log(10)

def mse2psnr(x):
    return -10.*torch.log(x)/__LOG10

# def get_features(dataset, model, device):
#     model.eval()
#     all_features = []
#     all_labels = []
#     dataset.two_params = True
#     dataloader = DataLoader(dataset, batch_size=1000, num_workers=8)
#     with torch.no_grad():
#         for x, _, params, _, _ in tqdm(dataloader):
#             z = model.enc(x.to(device))
#             all_features.append(z)

#             #params = (params + 4 * np.pi) % (2 * np.pi)
#             params[ params > np.pi ] = params[ params > np.pi ] - 2 * np.pi
#             all_labels.append(params)

#     return torch.cat(all_features).cpu().numpy(), \
#           torch.cat(all_labels).cpu().numpy()

def get_features(dataset, model, device, invariance=False):
    model.eval()
    all_features = []
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=1000, num_workers=8)
    with torch.no_grad():
        for x, _, y in tqdm(dataloader):
            x = x[0]
            z = model.enc(x.to(device))
            if invariance:
                z = model.amplitudes(z)
            z = z.flatten(start_dim=1)
            all_features.append(z)
            all_labels.append(y)

    return torch.cat(all_features).cpu().numpy(), \
           torch.cat(all_labels).cpu().numpy()

def linear_probe(dataset_train, dataset_test, model, device, n_samples=None, invariance=False, C=1.0):
    # Calculate the image features
    model.to(device)
    if n_samples is not None:
        dataset_train = torch.utils.data.Subset(dataset_train, range(n_samples))
    train_features, train_labels = get_features(dataset_train, model, device, invariance)
    test_features, test_labels = get_features(dataset_test, model, device, invariance)

    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features  = scaler.transform(test_features)

    #regressor = LinearRegression()
    #regressor.fit(train_features, train_labels)
    #predictions = regressor.predict(test_features)
    #rmse = np.sqrt(np.mean((test_labels - predictions) ** 2, axis=0))

    classifier = LogisticRegression(random_state=0, C=C, max_iter=1000)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    return accuracy

def distance_histogram(dataset, model, device, space='latent', mode=0, 
                       classwise=False, num_angles=72, tqdm=True,
                       metric=['l2', 'phase_sim']):
    '''
    mode = 0 --> search rotation, fix color
    mode = 1 --> search color, fix rotation
    '''
    model.to(device).eval()
    dataloader = DataLoader(dataset, batch_size=1000, num_workers=8)
    mins = []
    ys = []
    with torch.no_grad():
        iterator = tqdm(dataloader) if tqdm else dataloader
        for x1, x2, params, y in iterator:
            x1, x2, params = x1.to(device), x2.to(device), params.to(device)
            angle = torch.zeros_like(params)

            z1 = model.enc(x1)
            z2 = model.enc(x2)
            rmses = []
            for i in range(num_angles):
                rad = i * 2 * torch.pi / num_angles - torch.pi
                angle[:, mode] = params[:, mode] + rad
                z2_pred = model.trans(z1, angle.unbind(dim=-1))
                if space == 'latent':
                    if metric == 'l2':
                        rmse = torch.mean((z2 - z2_pred) ** 2, dim=-1)
                    elif metric == 'normalizedl2':
                        z2 = model.normalize_amplitude(z2)
                        z2_pred = model.normalize_amplitude(z2_pred)
                        rmse = F.mse_loss(z2, z2_pred, reduction='none').mean(dim=1)[:, :4].mean(dim=-1)
                    elif metric == 'phase_sim':
                        rmse = -model.phase_sim(z2_pred, z2).mean(dim=1)[:, mode]
                else:
                    x2_pred = model.decoder(z2_pred)
                    rmse = torch.mean((x2 - x2_pred) ** 2, dim=(1, 2, 3))
                rmses.append(rmse)
            min_rmse = torch.stack(rmses).argmin(dim=0)
            mins.append(min_rmse)
            ys.append(y)
    
    mins = torch.cat(mins).detach().cpu().numpy()
    mins = mins * (360 // num_angles) - 180
    op = ['rotation', 'color'][mode]
    if classwise:
        ys = torch.cat(ys)
        for i in range(10):
            plt.title(f'{dataset.__class__.__name__}/{space}/{op}/class{i}')
            plt.hist(mins[ys == i], bins=num_angles, range=(-180,180))
            plt.show()    
    else:
        plt.title(f'{dataset.__class__.__name__}/{space}/{op}')
        plt.hist(mins, bins=num_angles, range=(-180,180))
        plt.show()    