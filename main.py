import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from utils import RegDataset, RegNN, rejection_rmse
from regression_model import inference
from uncertainty_score import perturbation_score, gradnorm_score, mc_dropout_score, datafree_kd_score, ensemble_score
import sys, os, csv

dname = sys.argv[1]
n_layers = int(sys.argv[2])
method = sys.argv[3]
seed = int(sys.argv[4])

# load dataset
dfile = 'data/%s.csv'%dname   
  
data = np.genfromtxt(dfile, delimiter=',')
scaler = StandardScaler()
data = scaler.fit_transform(data)
    
X = data[:,:-1]
Y = data[:,-1:]
dim_x = X.shape[1]
    
print('-- dataset size: %d, no.features: %d'%(len(X), dim_x))
    
_, X_tst, _, Y_tst = train_test_split(X, Y, test_size=None, train_size=5000, random_state=seed)
test_set = RegDataset(X_tst, Y_tst)
test_loader = DataLoader(dataset=test_set, batch_size=500, shuffle=False)
    
# load the regression network
model_path = 'model/nn_%s_%d_%d.pt'%(dname, seed, n_layers)
net = RegNN(dim_x, n_layers = n_layers).cuda()
net.load_state_dict(torch.load(model_path))

# prediction with the network
Y_tst_hat = inference(net, test_loader).flatten()
Y_tst = Y_tst.flatten()   

# calcuate the uncertainty scroe. 
# choose one of the surrogates: 'input_perturbation', 'gradient_norm', 'mc_dropout', 'datafree_kd', 'ensemble'
if not os.path.exists('./student'):
    os.mkdir('./student')

if method == 'input_perturbation':
    uncertainty = perturbation_score(net, test_loader)

elif method == 'gradient_norm':
    uncertainty = gradnorm_score(net, X_tst)

elif method == 'mc_dropout':
    uncertainty = mc_dropout_score(net, test_loader)
    
elif method == 'datafree_kd':
    uncertainty = datafree_kd_score(net, test_loader, dname, seed, n_layers)
    
elif method == 'ensemble':
    uncertainty = ensemble_score(net, test_loader, X_tst, dname, seed, n_layers)

print('method: ', method)    
print('uncertainty score: ', uncertainty)

# calculate rejected RMSE with the uncertainty score & save the result
SE = (Y_tst - Y_tst_hat) ** 2

if not os.path.isfile('./result.csv'):
    with open('./result.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dname', 'seed', 'n_layers', 'method', 'rej_0', 'rej_5', 'rej_10', 'rej_20', 'rej_30', 'rej_40', 'rej_50'])

with open('./result.csv', 'a', newline='') as f:
    w = csv.writer(f)
    RMSE = rejection_rmse(SE, uncertainty)
    print(RMSE)
    w.writerow([dname, seed, n_layers, method]+RMSE)