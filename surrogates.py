import numpy as np
import torch
from utils import inference, datafree_kd

def perturbation_score(net, X_tst, perturbation_std=0.05, T=10):

    score = np.mean([(inference(net, X_tst, use_input_perturbation = True, perturbation_std=perturbation_std) - inference(net, X_tst)) ** 2 for _ in range(T)], 0)
    
    return score

def grad_norm_score(net, X_tst):

    cuda = torch.device('cuda:0')

    score = []
    
    for i in range(len(X_tst)):
    
        x = torch.FloatTensor(X_tst[i:i+1]).to(cuda)
        x.requires_grad = True
        y = net(x)
        y.backward(retain_graph=True)
        
        grad_norm = np.mean(np.square(x.grad.cpu().numpy()))
        
        score.append(grad_norm)
        
    return np.array(score)

def mc_dropout_score(net, X_tst, T=10):
    
    score = np.mean([(inference(net, X_tst, use_MC_dropout = True) - inference(net, X_tst)) ** 2 for _ in range(T)], 0)
    
    return score

def datafree_kd_score(net, X_tst, T=10):

    score = np.mean([(datafree_kd(net, X_tst) - inference(net, X_tst)) ** 2 for _ in range(T)], 0)
    
    return score
