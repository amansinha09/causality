# original_shap.py

import time
import math
import itertools
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from torch.utils.data import Dataset, DataLoader

from utils import *
from data import timedata
import gru


######################
'''
Add any shap simulator with following arguments

Input : 
    s : subset s 
    sui: subset s U i
    X_train: type(dataframe)
    Y_trains: type(dataframe)
    X_test: type(dataframe)
    Y_tests: type(dataframe)
    model: modelname 

Output:
    Sum over all difference between prediction_sui and prediction_s

'''
######################

def blah(s,sui, X_train, Y_trains, X_test, Y_tests, model):
    # s and sui are set of feature indexes
    # copy dataset to two set of features
    # train two models
    # shuffle keep same
    # for any sample in test
    Xs_train = X_train[['feat'+str(i) for i in s]]
    Xsui_train = X_train[['feat'+str(i) for i in sui]]
    #print(Xs_train.head(), Xsui_train.head())
    #return 0
    
    Xs_test = X_test[['feat'+str(i) for i in s]]
    Xsui_test = X_test[['feat'+str(i) for i in sui]]

    if model == 'knn':
        clfs = KNeighborsRegressor(n_neighbors=2)#RandomForestClassifier(max_depth=2, random_state=0)
        clfsui = KNeighborsRegressor(n_neighbors=2)#RandomForestClassifier(max_depth=2, random_state=0)
    elif model =='rf':
        clfs = RandomForestRegressor(max_depth=2, random_state=0)# s
        clfsui = RandomForestRegressor(max_depth=2, random_state=0)# sui
    
    if model != 'gru':
        clfs.fit(Xs_train, Y_trains)
        clfsui.fit(Xsui_train, Y_trains)

        fsui = clfsui.predict(Xsui_test)
        fs = clfs.predict(Xs_test)
    else:
        #print('running grus...')
        #print(Xs_train.head(),'\n', X_test.head())
        params = {'lr': 0.001, 'hidden_dim':256,'epochs': 5,}

        traindfs = pd.concat([Xs_train, Y_trains], axis=1)
        traindfsui = pd.concat([Xsui_train, Y_trains], axis=1)
        
        testdfs = pd.concat([Xs_test, Y_tests], axis=1)
        testdfsui = pd.concat([Xsui_test, Y_tests], axis=1)
        
        traindataloaders = DataLoader(timedata(traindfs, Xs_train.shape[1]), 32, shuffle=True, drop_last=True)
        traindataloadersui = DataLoader(timedata(traindfsui, Xsui_train.shape[1]), 32, shuffle=True, drop_last=True)
        
        testloaders = DataLoader(timedata(testdfs, Xs_test.shape[1]), 32, shuffle=True, drop_last=True)
        testloadersui = DataLoader(timedata(testdfsui, Xsui_test.shape[1]), 32, shuffle=True, drop_last=True)
        
        #print('==>',traindfs.shape[1]-1, traindfsui.shape[1]-1)
        models = gru.train(traindataloaders, params,inpdim=traindfs.shape[1]-1, num=2)
        modelsui = gru.train(traindataloadersui, params,inpdim=traindfsui.shape[1]-1,num= 2)
        
        fsui = gru.evaluate(models, testloaders)
        fs = gru.evaluate(modelsui, testloadersui)
        
    return sum(np.array(fsui) - np.array(fs))

def simulate_shap(nfeat, df, Ys, model='rf'):
    #nfeat = 3
    assert nfeat <= len(df.columns)

    X = df.iloc[:, 0:nfeat]
    print(X.head())
    shaps = np.zeros(nfeat)
    stuff = [i for i in range(nfeat)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Ys, test_size=0.15, random_state=42)
    
    print(f"No of features: {nfeat}")
    print(f"Shape of X: {X_train.shape}")
    print(f'Using model: {model}')
    
    subsets = [] 
    for L in range(0, len(stuff)+1):
        for subset in itertools.combinations(stuff, L):
            subsets.append(set(subset))

    times = np.zeros(nfeat)

    t1 = time.time()
    f = {i for i in range(nfeat)}
    for i, sh in enumerate(shaps):
        print(f"Current feat: {i+1}th")

        for s in subsets:

            if not (i in s) and len(s)>0:

                sui = s | {i}
                #print(s, sui)
                # absolute values 
                shaps[i] += k(s,f)* blah(s,sui,X_train, y_train, X_test, y_test, model=model)
                #abs(k(s,f)* blah(s,sui,X_train, y_train, X_test, y_test))
        t2 = time.time()
        times[i] = t2 - t1
        t1 = t2
        
        
    
    return times, shaps