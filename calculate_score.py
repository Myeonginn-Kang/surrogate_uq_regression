import numpy as np
import torch
from utils import set_dropout
from surrogates import perturbation_score, grad_norm_score, mc_dropout_score, datafree_kd_score


def calculate_score(net, X_tst, method):
     
    if method == 'input_perturbation':
        uncertainty = perturbation_score(net, X_tst)
        uncertainty = np.sqrt(uncertainty)
        
    elif method == 'gradient_norm':
        uncertainty = grad_norm_score(net, X_tst)
        uncertainty = np.sqrt(uncertainty)
        
    elif method == 'mc_dropout':
        set_dropout(net, drop_rate=0.05)
        uncertainty = mc_dropout_score(net, X_tst)
        uncertainty = np.sqrt(uncertainty)
        
    elif method == 'datafree_kd':
        uncertainty = datafree_kd_score(net, X_tst)
        uncertainty = np.sqrt(uncertainty)
        
    elif method == 'ensemble':
        uncertainty1 = perturbation_score(net, X_tst)      
        uncertainty2 = grad_norm_score(net, X_tst)
        set_dropout(net, drop_rate=0.05)
        uncertainty3 = mc_dropout_score(net, X_tst)
        set_dropout(net, drop_rate=0)
        uncertainty4 = datafree_kd_score(net, X_tst)
        uncertainty = np.sqrt(uncertainty1)+np.sqrt(uncertainty2)
        + np.sqrt(uncertainty3) + np.sqrt(uncertainty4)

    return uncertainty
