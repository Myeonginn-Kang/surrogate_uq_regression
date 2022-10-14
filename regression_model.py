import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

def training(net, train_loader, val_loader, model_path, max_epochs = 500, cuda = torch.device('cuda:0')):

    cuda = torch.device('cuda:0')

    loss_fn = nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    val_log = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        
        # training
        net.train()
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
        print('--- val monitoring, processed %d, current RMSE %.3f, best RMSE %.3f' %(val_loader.dataset.__len__(), val_log[epoch], np.min(val_log[:epoch + 1])))
    
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        if np.argmin(val_log[:epoch + 1]) <= epoch - 20: break
    
    net.load_state_dict(torch.load(model_path))
    
    print('training terminated at epoch %d' %epoch)
    
    return net
    

def inference(net, test_loader, cuda = torch.device('cuda:0')):

    cuda = torch.device('cuda:0')
    
    # inference
    net.eval()

    y_hat = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            batch_x = batchdata[0]            
            batch_x = batch_x.to(cuda)

            batch_y_hat = net(batch_x).cpu().numpy()
            
            y_hat.append(batch_y_hat)
            
    y_hat = np.vstack(y_hat).flatten()

    return y_hat
    