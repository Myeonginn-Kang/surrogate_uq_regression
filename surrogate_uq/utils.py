import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop


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
        

def training(net, train_loader, val_loader, model_path, max_epochs = 500):

    cuda = torch.device('cuda:0')

    loss_fn = nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    val_log = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):
    
            batch_x, batch_y = batchdata
            batch_x, batch_y = batch_x.to(cuda), batch_y.to(cuda)
            
            batch_y_hat = net(batch_x)
    
            loss = loss_fn(batch_y_hat, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # validation
        net.eval()
        val_loss, val_cnt = 0, 0
        with torch.no_grad():
            for batchidx, batchdata in enumerate(val_loader):
            
                batch_x = batchdata[0].to(cuda)
                batch_y = batchdata[1].numpy()
    
                batch_y_hat = net(batch_x).cpu().numpy()
    
                loss = (batch_y_hat - batch_y) ** 2
                
                val_loss += np.sum(loss)
                val_cnt += len(loss)
    
        val_log[epoch] = np.sqrt(val_loss/val_cnt)
        #print('--- val monitoring, processed %d, current RMSE %.3f, best RMSE %.3f' %(val_loader.dataset.__len__(), val_log[epoch], np.min(val_log[:epoch + 1])))
    
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        if np.argmin(val_log[:epoch + 1]) <= epoch - 20: break
    
    net.load_state_dict(torch.load(model_path))
    
    best_RMSE = np.min(val_log[:epoch + 1])
    best_epoch = np.argmin(val_log[:epoch + 1])
    print('best RMSE %.3f at epoch %d'%(best_RMSE, best_epoch))
    
    summary = [best_RMSE, best_epoch]
    
    return net, summary
    

def inference(net, test_loader, use_input_perturbation = False, perturbation_std = 0.05, use_MC_dropout = False):

    def MC_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    cuda = torch.device('cuda:0')
    
    # inference
    net.eval()
    if use_MC_dropout: MC_dropout(net)

    y_hat = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            batch_x = batchdata[0]
            if use_input_perturbation: batch_x += (perturbation_std * np.random.randn(batch_x.shape[0], batch_x.shape[1])).astype(np.float32)
            
            batch_x = batch_x.to(cuda)

            batch_y_hat = net(batch_x).cpu().numpy()
            
            y_hat.append(batch_y_hat)
            
    y_hat = np.vstack(y_hat).flatten()

    return y_hat
    

def set_dropout(model, drop_rate=0.05):

    for name, child in model.named_children():
    
        if isinstance(child, nn.Dropout):
            child.p = drop_rate
            
        set_dropout(child, drop_rate=drop_rate)

    
def datafree_kd(net, test_loader, dim_z = 50, beta = 1e-5, gamma = 1e-5, batch_size = 50, n_iter = 2000, n_g = 1, n_s = 10):
    cuda = torch.device('cuda:0')

    dim_x = net.state_dict()['predict.0.weight'].shape[1]

    generator = GenNN(dim_x).to(cuda)
    student = RegNN(dim_x, n_layers=1, dim_h=25).to(cuda)
    
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
            
            S_loss = torch.mean( alpha * torch.square(batch_ytg - batch_ysg) + (1 - alpha) * torch.square(batch_ytp - batch_ysp) )
            
            optimizerS.zero_grad()
            S_loss.backward()
            optimizerS.step()


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