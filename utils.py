import numpy as np
import torch
import torch.nn as nn

class RegDataset():

    def __init__(self, X, Y):
    
        self.X = X
        self.Y = Y

    def __len__(self):
    
        return len(self.X)

    def __getitem__(self, idx):
    
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
    
        return x, y


class RegNN(nn.Module):

    def __init__(self, dim_x, n_layers=3, dim_h=100, prob_dropout=0):
        
        super(RegNN, self).__init__()

        layers = [nn.Linear(dim_x, dim_h), nn.Tanh(), nn.Dropout(prob_dropout)]
        for _ in range(n_layers-1):
            layers += [nn.Linear(dim_h, dim_h), nn.Tanh(), nn.Dropout(prob_dropout)]
        
        layers += [nn.Linear(dim_h, 1)]

        self.predict = nn.Sequential(*layers)                           
                               
    def forward(self, x):

        y_hat = self.predict(x)

        return y_hat


class GenNN(nn.Module):

    def __init__(self, dim_x, dim_z=50, dim_h=500):
        
        super(GenNN, self).__init__()

        layers = [nn.Linear(dim_z, dim_h), nn.Tanh(), nn.Linear(dim_h, dim_x)]

        self.predict = nn.Sequential(*layers)                           
                               
    def forward(self, z):

        x = self.predict(z)

        return x
        
def set_dropout(model, drop_rate=0.05):

    for name, child in model.named_children():
    
        if isinstance(child, nn.Dropout):
            child.p = drop_rate
            
        set_dropout(child, drop_rate=drop_rate)

    
def rejection_rmse(squared_error, score):

    arg_id = np.argsort(score) # the smaller the better
    squared_error = squared_error[arg_id]
    
    rmse_list = []
    for rate in [0, 5, 10, 20, 30, 40, 50]:
        
        cnt = int(len(arg_id) * (100-rate)/100)
        rmse_list.append(np.sqrt(np.mean(squared_error[:cnt])))
    
    return rmse_list

def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass