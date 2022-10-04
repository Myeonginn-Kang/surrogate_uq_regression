import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import csv
import torch
from torch.utils.data import DataLoader
from utils import RegDataset, RegNN, training


def train_model(dname, seed, n_layers, batch_size = 50, max_epochs = 300):
 
    dfile = 'data/%s.csv'%dname    
    model_path = 'model/nn_%s_%d_%d.pt'%(dname, seed, n_layers)
    
    data = np.genfromtxt(dfile, delimiter=',')
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    X = data[:,:-1]
    Y = data[:,-1:]
    dim_x = X.shape[1]
    
    print(dname, seed, n_layers)
    print('-- dataset size: %d, n. features: %d'%(len(X), dim_x))
    
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=None, train_size=5000, random_state=seed)
    train_set = RegDataset(X_trn, Y_trn)
    test_set = RegDataset(X_tst, Y_tst)
    
    print('-- trn/tst: %d/%d'%(train_set.__len__(), test_set.__len__()))
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size * 10, shuffle=False)

    net = RegNN(dim_x, n_layers).cuda()
    net, summary = training(net, train_loader, test_loader, model_path)
    
    return [dname, seed, n_layers] + summary

    
with open('./model_train_result.csv', 'w') as f:
    wr = csv.writer(f)
    wr.writerow(['dname', 'seed', 'n_layers', 'best_RMSE', 'best_epoch'])
    
    for dname in ['bikesharing','compactiv','cpusmall','ctscan','indoorloc','mv','pole','puma32h','telemonitoring']:
        for seed in range(10):
            for n_layers in [1, 2, 3]:
            
                wr.writerow(train_model(dname,seed,n_layers))