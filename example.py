import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from utils import RegNN, inference, rejection_rmse
from calculate_score import calculate_score

# load dataset
dfile = 'data/bikesharing.csv'   
  
data = np.genfromtxt(dfile, delimiter=',')
scaler = StandardScaler()
data = scaler.fit_transform(data)
    
X = data[:,:-1]
Y = data[:,-1:]
dim_x = X.shape[1]
    
print('-- dataset size: %d, no.features: %d'%(len(X), dim_x))
    
_, X_tst, _, Y_tst = train_test_split(X, Y, test_size=None, train_size=5000, random_state=0)

# load the regression network
model_path = 'model/nn_bikesharing_0_1.pt'
net = RegNN(dim_x, n_layers = 1).cuda()
net.load_state_dict(torch.load(model_path))

# prediction with the network
Y_tst_hat = inference(net, X_tst).flatten()
Y_tst = Y_tst.flatten()   

# calcuate the uncertainty scroe. 
# choose one of the surrogates: 'input_perturbation', 'gradient_norm', 'mc_dropout', 'datafree_kd', 'ensemble'
uncertainty = calculate_score(net, X_tst, method = 'input_perturbation')
print('uncertainty score: ', uncertainty)

# calculate rejected RMSE with the uncertainty score
SE = (Y_tst - Y_tst_hat) ** 2
RMSE = rejection_rmse(SE, uncertainty, rate=0.05)
print('RMSE: ', RMSE)