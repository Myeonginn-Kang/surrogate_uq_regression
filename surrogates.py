import numpy as np
import torch
from utils import MC_dropout, RegNN, GenNN
import torch
import torch.nn as nn
from torch.optim import RMSprop

def input_perturbation(net, test_loader, perturbation_std = 0.05, cuda = torch.device('cuda:0')):
   
    net.eval()

    y_hat = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):        
            batch_x = batchdata[0]
            batch_x += (perturbation_std * np.random.randn(batch_x.shape[0], batch_x.shape[1])).astype(np.float32)
            batch_x = batch_x.to(cuda)
            batch_y_hat = net(batch_x).cpu().numpy()            
            y_hat.append(batch_y_hat)            
    y_hat = np.vstack(y_hat).flatten()

    return y_hat

def gradient_norm(net, X_tst, cuda = torch.device('cuda:0')):

    score = []
    
    for i in range(len(X_tst)):
    
        x = torch.FloatTensor(X_tst[i:i+1]).to(cuda)
        x.requires_grad = True
        y = net(x)
        y.backward(retain_graph=True)
        
        grad_norm = np.mean(np.square(x.grad.cpu().numpy()))
        
        score.append(grad_norm)
        
    return np.array(score)

def mc_dropout(net, test_loader, cuda = torch.device('cuda:0')):
   
    net.eval()
    MC_dropout(net)

    y_hat = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):        
            batch_x = batchdata[0]            
            batch_x = batch_x.to(cuda)
            batch_y_hat = net(batch_x).cpu().numpy()            
            y_hat.append(batch_y_hat)            
    y_hat = np.vstack(y_hat).flatten()

    return y_hat

def datafree_kd(net, test_loader, student_path, student_training = False, dim_z = 50, beta = 1e-5, gamma = 1e-5, batch_size = 50, n_iter = 2000, n_g = 1, n_s = 10, cuda = torch.device('cuda:0')):

    dim_x = net.state_dict()['predict.0.weight'].shape[1]

    generator = GenNN(dim_x).to(cuda)
    student = RegNN(dim_x, n_layers=1, dim_h=25).to(cuda)
    
    if student_training == True:
    
        optimizerG = RMSprop(generator.parameters(), lr=1e-3, weight_decay=1e-5)
        optimizerS = RMSprop(student.parameters(), lr=1e-3, weight_decay=1e-5)

        generator.train()
        student.train()
        for it in range(n_iter):

            alpha = 1 - (it+1)/n_iter

            # generator training
            for _ in range(n_g):

                batch_z = torch.randn(batch_size, dim_z).to(cuda)
                batch_xg = generator(batch_z)
                batch_ytg = net(batch_xg)
                batch_ysg = student(batch_xg)

                G_loss = torch.mean( - torch.square(batch_ytg - batch_ysg) + beta * torch.sum(torch.square(batch_xg)) + gamma * torch.square(batch_ysg) )

                optimizerG.zero_grad()
                G_loss.backward()
                optimizerG.step()

            # student training
            for _ in range(n_s):

                batch_z = torch.randn(batch_size, dim_z).to(cuda)
                batch_xg = generator(batch_z)
                batch_ytg = net(batch_xg)
                batch_ysg = student(batch_xg)

                batch_xp = torch.randn(batch_size, dim_x).to(cuda)
                batch_ytp = net(batch_xp)
                batch_ysp = student(batch_xp) 

                S_loss = torch.mean(alpha * torch.square(batch_ytg - batch_ysg) + (1 - alpha) * torch.square(batch_ytp - batch_ysp) )

                optimizerS.zero_grad()
                S_loss.backward()
                optimizerS.step()
        torch.save(student.state_dict(), student_path) 
    
    else:
        student.load_state_dict(torch.load(student_path))
        
    # inference
    student.eval()
    
    y_hat = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            batch_x = batchdata[0]
            batch_x = batch_x.to(cuda)

            batch_y_hat = student(batch_x).cpu().numpy()
            
            y_hat.append(batch_y_hat)
            
    y_hat = np.vstack(y_hat).flatten()
    
    return y_hat
