# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:42:18 2022

@author: liuxiang
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import scipy as sp




Pre = './s645/'

def get_combined_feature():
    atom_typ = [ 'C','N','O','CN','CO','NO' ]
    site_typ = [ 'mutate2','wild2' ]
    tor_typ = [ '2_h0','3_h0','99_h0','99_h1','100_h0','100_h1' ]
    
    for i in range(2):
        for j in range(6):
            for k in range(6):
                filename = Pre + 'feature/separate-feature/' + site_typ + '_' + atom_typ + '_' + tor_typ + '.npy'
                d1 = np.load(filename)
                filename = Pre + 'feature.X_aux.csv'
                d2 = np.loadtxt(filename,delimiter=',')
                d = np.hstack((d1,d2))
                filename = Pre + 'feature/separate-feature/' + site_typ + '_' + atom_typ + '_' + tor_typ + '_aux.npy'
                np.save(filename,d)
                

def normalize(X):
    t = X.shape
    Y = X
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    for i in range(t[0]):
        for j in range(t[1]):
            if std[j]!=0:
                Y[i][j] = (Y[i][j]-mean[j])/std[j]
            else:
                if Y[i][j]!=0:
                    Y[i][j] = 1
    return Y


def gradient_boosting(X_train,Y_train,X_test):
    
    params={'n_estimators': 4000, 'max_depth': 6, 'min_samples_split': 2,
                'learning_rate': 0.01, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train,Y_train)
    
    a_predict = regr.predict(X_test)
    res = []
    for i in range(len(a_predict)):
        res.append(a_predict[i])
    return res


def get_each_feature_for_stacking(typ,typ2):
    filename = Pre + 'feature/separate-feature/' + typ + '.npy'
    feature_matrix = np.load(filename)
    
    #pre = 'E:\\hom_complex\\s645_feature\\'
    filename1 = Pre + 'feature/nonbinder_10crossvalidation_index.txt'
    f = open(filename1)
    pre_index = f.read()
    index = eval(pre_index)
    f.close()
    
    filename3 = Pre + 'feature/label.csv'
    label = np.loadtxt(filename3,delimiter=',')
    
    feature_matrix = normalize(feature_matrix)
    feature_shape = feature_matrix.shape
    res = []
    for i in range(10):
        subindex = index[i]
        temp = len(subindex)
        test_feature = np.zeros((temp,feature_shape[1]))
        pre_test_label = []
        
        train_feature = np.zeros((645-27-temp,feature_shape[1])) # 645-27 or 1131 or 645
        pre_train_label = []
        test_c = 0
        train_c = 0
        for j in range(10):
            if j==i:
                for k in index[j]:    
                    test_feature[test_c,:] = feature_matrix[k,:]
                    pre_test_label.append(label[k])
                    test_c = test_c + 1
            else:
                for k in index[j]:
                    train_feature[train_c,:] = feature_matrix[k,:]
                    pre_train_label.append(label[k])
                    train_c = train_c + 1
        train_label = np.array(pre_train_label)
        test_label = np.array(pre_test_label)
        print(train_feature.shape,train_label.shape,test_feature.shape,test_label.shape)
        temp_res = gradient_boosting(train_feature,train_label,test_feature)
        for value in temp_res:
            res.append(value)
        
        #print(temp_res)
    #final_res = np.array(res)
    final_res = np.zeros((645,1)) # 645 or 1131
    all_index = []
    for i in range(10):
            for value in index[i]:
                all_index.append(value)
    for i in range(len(all_index)):
        real_index = all_index[i]
        final_res[real_index,0] = res[i]
    
    
    filename = Pre + 'gbt-feature/gbt_' + typ2 + '_' + typ + '.npy'
    np.save(filename,final_res)
    
def get_combine(N):
    filename = './s645/gbt-feature/gbt_' + N + '_mutate2_C_2_h0_aux.npy'
    d1 = np.load(filename)
    for typ1 in site_typ:
        for typ2 in atom_typ:
            for typ3 in tor_typ:
                if typ1=='mutate2' and typ2=='C' and typ3=='2_h0':
                    continue
                name = site_typ[i] + '_' + atom_typ[j] + '_' + tor_typ[k] + '_aux'
                filename = './s645/gbt-feature/gbt_' + N + '_' + name + '_.npy'
                d2 = np.load(filename)
                d1 = np.hstack((d1,d2))
    filename = './s645/gbt-feature/' + N + 'gbt.npy'
    np.save(filename,d1)
    
get_combined_feature()
atom_typ = [ 'C','N','O','CN','CO','NO' ]
site_typ = [ 'mutate2','wild2' ]
tor_typ = [ '2_h0','3_h0','99_h0','99_h1','100_h0','100_h1' ]
for N in range(10):
    for i in range(2):
        for j in range(6):
            for k in range(6):
                name = site_typ[i] + '_' + atom_typ[j] + '_' + tor_typ[k] + '_aux'
                main(name,str(N))
                #print(name,typ,'ok')  
    get_combine(str(N))
    
    
    
    
    
    
    
    
    
    
    