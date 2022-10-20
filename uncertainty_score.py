import numpy as np
from utils import set_dropout
from surrogates import input_perturbation, gradient_norm, mc_dropout, datafree_kd
from regression_model import inference
import os

def perturbation_score(net, test_loader, perturbation_std=0.05, T=10):

    score = np.mean([(input_perturbation(net, test_loader, perturbation_std=perturbation_std) - inference(net, test_loader)) ** 2 for _ in range(T)], 0)
    
    return score

def gradnorm_score(net, X_tst):
    
    score = gradient_norm(net, X_tst)
    
    return score

def mc_dropout_score(net, test_loader, T=10):
    
    score = np.mean([(mc_dropout(net, test_loader) - inference(net, test_loader)) ** 2 for _ in range(T)], 0)
    
    return score

def datafree_kd_score(net, test_loader, dname, seed, n_layers, T=10):

    student_training = False
    if not os.path.isfile(('./student/student_%s_%d_%d_0.pt')%(dname, seed, n_layers)):
        student_training = True
    score = np.mean([(datafree_kd(net, test_loader, student_path='./student/student_%s_%d_%d_%d.pt'%(dname, seed, n_layers,T),student_training=student_training) 
    - inference(net, test_loader)) ** 2 for _ in range(T)], 0)

    return score

def ensemble_score(net, test_loader, X_tst, dname, seed, n_layers, student_training=False, T = 10):

    uncertainty1 = perturbation_score(net, test_loader)      
    uncertainty2 = gradnorm_score(net, X_tst)
    set_dropout(net, drop_rate=0.05)
    uncertainty3 = mc_dropout_score(net, test_loader)
    set_dropout(net, drop_rate=0)
    student_training = False
    if not os.path.isfile(('./student/student_%s_%d_%d_0.pt')%(dname, seed, n_layers)):
        student_training = True
    uncertainty4 = np.mean([(datafree_kd(net, test_loader, student_path='./student/student_%s_%d_%d_%d.pt'%(dname, seed, n_layers,T),student_training=student_training) 
    - inference(net, test_loader)) ** 2 for _ in range(T)], 0)
    uncertainty = np.sqrt(uncertainty1) + np.sqrt(uncertainty2) + np.sqrt(uncertainty3) + np.sqrt(uncertainty4)

    return uncertainty