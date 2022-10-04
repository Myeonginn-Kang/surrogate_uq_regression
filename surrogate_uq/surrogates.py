import numpy as np
import torch
from utils import inference, datafree_kd

def perturbation_score(net, test_loader,Y_tst_hat, perturbation_std=0.05):

    score = np.mean([(inference(net, test_loader, use_input_perturbation = True, perturbation_std) - Y_tst_hat) ** 2 for _ in range(T)], 0)
    
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

def mc_dropout_score(net, test_loader, Y_tst_hat):
    
    score = np.mean([(inference(net, test_loader, use_MC_dropout = True) - Y_tst_hat) ** 2 for _ in range(T)], 0)
    
    return score

def datafree_kd_score(net, test_loader, Y_tst_hat):

    score = np.mean([(datafree_kd(net, test_loader) - Y_tst_hat) ** 2 for _ in range(T)], 0)
    
    return score
