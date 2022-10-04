import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from utils import RegDataset, RegNN, inference, grad_norm, set_dropout, datafree_kd
from surrogate import perturbation_score, grad_norm_score, mc_dropout_score, datafree_kd_score


def reject_with_surrogate(dname, seed, n_layers, method, batch_size = 50, T = 10):

    def rejection_rmse(squared_error, score):
    
        arg_id = np.argsort(score) # the smaller the better
        squared_error = squared_error[arg_id]
        
        rmse_list = []
        for rate in [0, 5, 10, 20, 30, 40, 50]:
            
            cnt = int(len(arg_id) * (100-rate)/100)
            rmse_list.append(np.sqrt(np.mean(squared_error[:cnt])))
        
        return rmse_list
 
 
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
    
    _, X_tst, _, Y_tst = train_test_split(X, Y, test_size=None, train_size=5000, random_state=seed)

    test_set = RegDataset(X_tst, Y_tst)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size * 10, shuffle=False)

    net = RegNN(dim_x, n_layers).cuda()
    net.load_state_dict(torch.load(model_path))
    Y_tst_hat = inference(net, test_loader).flatten()
    
    Y_tst = Y_tst.flatten()    
    SE = (Y_tst - Y_tst_hat) ** 2
    
    print('RMSE:', np.sqrt(np.mean(SE)))
    
    if method == 'input_perturgation':
        uncertainty = perturbation_score(net, test_loader, Y_tst_hat, perturbation_std=0.05)
        uncertainty = np.sqrt(uncertainty)
        
    elif method == 'gradient_norm:
        uncertainty = grad_norm_score(net, X_tst)
        uncertainty = np.sqrt(uncertainty)
        
    elif method == 'mc_dropout':
        set_dropout(net, drop_rate=0.05)
        uncertainty = mc_dropout_score(net, test_loader, Y_tst_hat)
        uncertainty = np.sqrt(uncertainty)
        
    elif method == 'datafree_kd':
        uncertainty = datafree_kd_score(net, test_loader, Y_tst_hat)
        uncertainty = np.sqrt(uncertainty)
        
    elif method == 'ensemble:
        uncertainty1 = perturbation_score(net, test_loader, Y_tst_hat, perturbation_std=0.05)      
        uncertainty2 = grad_norm_score(net, X_tst)
        set_dropout(net, drop_rate=0.05)
        uncertainty3 = mc_dropout_score(net, test_loader, Y_tst_hat)
        set_dropout(net, drop_rate=0)
        uncertainty4 = datafree_kd_score(net, test_loader, Y_tst_hat)
        uncertainty = np.sqrt(uncertainty1)/np.std(uncertainty1) +np.sqrt(uncertainty2)/np.std(uncertainty2) 
        + np.sqrt(uncertainty3)/np.std(uncertainty3) + np.sqrt(uncertainty4)/np.std(uncertainty4)
    
    rmse = rejection_rmse(SE, uncertainty)

    return rmse