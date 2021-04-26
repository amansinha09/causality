#data.py
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

class timedata(Dataset):
    def __init__(self, df, nfeatures):
            self.X = df.iloc[:,:nfeatures].values.astype(np.float32)
            self.Y = df.iloc[:,nfeatures:].values.astype(np.float32)
            
    def __len__(self): return len(self.Y)
    
    def __getitem__(self, idx):
        return [np.expand_dims(self.X[idx], axis=0), self.Y[idx]]


class TSElement(object):
    def __init__(self,dt, val=0):
        self.dt = dt
        self.val = val
        
    def __str__(self):
        string = []
        
        id_dt= self.dt
        string.append(f'Datetime: {self.dt}')
        
        id_val = self.val
        string.append(f'Value : {self.val}')
        
        return string
    
class TimeSeriesData:
    def __init__(self, df, name, h=3,y=1,k=1):
        self.name = name
        self.h = h
        self.y = y
        self.instances = None
        self.df = df[::k]
        self.k = k
        
    def __len__(self):
        return len(self.df)
    
    def prepare_data(self):
        l = list(zip(self.df['datetime'],self.df['val']))
        # list of list of X,y
        #self.instances  = [[TSElement(*o) for o in l[i:i+self.h]] for i,e in enumerate(l) if (i+self.h) < len(df)]
        self.instances  = [[TSElement(*o) for o in l[i:i+self.h+self.y]] for i,e in enumerate(l) if (i+self.h+self.y) < len(self.df)]
        
        X,Y = [],[]
        for instance in self.instances:
            X.append([ins.val for ins in instance[:self.h]])
            Y.append([ins.val for ins in instance[self.h:]])
        featnames = [f'feat{i}' for i in range(self.h)]
        ynames = [f'Y{i}' for i in range(self.y)]

        xdf = pd.DataFrame(X, columns=featnames)
        ydf = pd.DataFrame(Y, columns=ynames)
        datadf = pd.concat([xdf, ydf], axis=1)
        datadf = datadf.dropna()
        self.df = datadf
        
        print('Size of Dataset:', len(self.df))
        
    
    def avg_baseline(self):
        # output sequence of length 1
        predictions = []
        truths = []
        
        for instance in self.instances:
            seq = instance[:self.h]
            truths.append(instance[self.h:])
            predictions.append(np.mean([ins.val for ins in seq]))
            
        loss = mean_squared_error(truths, predictions)
        
        return truths, predictions, loss
    
    def knn_reg_baseline(self):
        
        predictions = []
        X,y = [],[]
        
        for instance in self.instances:
            X.append([ins.val for ins in instance[:self.h]])
            y.append(instance[self.h:].val)
        
        
        X_train, X_test, y_train, truths = train_test_split(X, y, test_size=0.33, random_state=42)
        
        neigh = KNeighborsRegressor(n_neighbors=2)
        neigh.fit(X_train, y_train)
        
        predictions = neigh.predict(X_test)
        
        loss = mean_squared_error(predictions, truths)
        
        return truths, predictions, loss