#gru.py
import time

import torch
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cpu')

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim, n_layer, bidir, dropout=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidir = bidir
        self.n_layer = n_layer
        
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.n_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        #print(x.shape, h.shape)
        out, h = self.gru(x, h)
        out  = self.fc(self.relu(out))
        return out, h
    
    def init_hidden(self, batch_size):
        wt = next(self.parameters()).data
        hidden = wt.new(self.n_layer, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

def train(trainloader ,params, inpdim,num=None):
    
    input_dim = inpdim
    #print(f"working with {inpdim} features")#nextitem(iter(trainloader))[0].shape[2]
    output_dim = 1
    n_layers = 2
    batch_size= 32
    
    model = GRUNet(input_dim, hidden_dim=params['hidden_dim'],
                   output_dim=output_dim, n_layer=n_layers, bidir=False)
    model.to(device)
    
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    if num:
        writer = SummaryWriter(f'runs/gru_experiment_{num}')
    model.train()
    epoch_times  = []
    p =0
    for epoch in range(params['epochs']):
        start = time.time()
        h = model.init_hidden(batch_size)
        avg_loss = 0
        counter = 0
        
        for x, label in trainloader:
            counter +=1
            p +=1
            h = h.data
            
            model.zero_grad()
            
            out, h = model(x.to(device).float(), h)
            #print(out.shape, label.shape)
            out = torch.squeeze(out, dim=-1)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            
            if counter % 200 == 0:
                print(f"Epoch : {epoch}, Step: {counter}/{len(trainloader)} ==> Avg Loss for epoch: {avg_loss/counter}")
                if num: writer.add_scalar('training_loss', avg_loss/counter, p)
        current_time = time.time()
        #print(f"Total time elapsed: {current_time-start} seconds")
        epoch_times.append(current_time- start)
    
    #print(f"Total Training Time: {sum(epoch_times)} seconds")
    return model



def evaluate(model, testdataloader):
    model.eval()
    outputs = []
    targets = []
    
    start = time.time()
    
    for x,y in testdataloader:
        
        h = model.init_hidden(x.shape[0])
        out, h = model(x.to(device), h)
        outputs.extend(out.cpu().detach().numpy().squeeze())
        targets.extend(y.numpy())
        
    #print(targets[0].shape, outputs[0].squeeze().shape)
    #for t,p in zip(targets, outputs): 
     #   print(t,p.squeeze())
    
    #print(f"Evaluation time {time.time()- start}")
    #smape = 0
    
    #for i in range(len(outputs)):
        #smape += np.mean(abs(outputs[i]-targets[i])/abs(outputs[i]+targets[i])/2)/len(outputs)
       # smape += abs(outputs[i]-targets[i])/abs(outputs[i]+targets[i])/2
    
    #smape /= len(outputs)
        
    #print(f"smape: {smape *100}")
    return outputs
            