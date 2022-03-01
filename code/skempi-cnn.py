# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import h5py
from tensorflow.keras.models import load_model
import time



class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        logs["test_loss"] = loss
        logs["test_pcc"] = acc
        print('Testing loss: {}, acc: {}'.format(loss, acc))


def normalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    length1 = X.shape[0]
    X_train_normed = X

    for i in range(0,length1):
        for j in range(0,X.shape[1]):
            for k in range(0, X.shape[2]):
                if std[j,k]!=0 :
                    X_train_normed[i,j,k] = (X_train_normed[i,j,k]-mean[j,k])/std[j,k]
    return X_train_normed


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def main(typ,N):
    grid_size = 0.25
    filtration = 10
    grid_number = int(filtration/grid_size)
    feature_number = 6
    
    pre = './skempi/feature/'
    filename1 = pre + '10crossvalidation_index.txt'
    f = open(filename1)
    pre_index = f.read()
    index = eval(pre_index)
    f.close()
    
    filename2 = pre + 'separate-feature/' + typ + '.npy'
    print('edge')
    feature_matrix = np.load(filename2)
    feature_shape = feature_matrix.shape
    
    filename3 = pre + 'label.csv'
    label = np.loadtxt(filename3,delimiter=',')
    #res1 = np.zeros((1131,800))
    #res2 = np.zeros((1131,10))
    res3 = np.zeros((1131,1))
    final_pcc = []
    for i in range(10):
        print('CV',i)
        test_length = len(index[i])
        test_feature = np.zeros((test_length,feature_shape[1]))
        pre_test_label = []
        
        train_feature = np.zeros((1131-test_length,feature_shape[1]))
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
        
        test_feature = test_feature.reshape((test_length,grid_number,feature_number))
        train_feature = train_feature.reshape((1131-test_length,grid_number,feature_number))
        test_feature = normalize(test_feature)
        train_feature = normalize(train_feature)
        
        saved_hist = []
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(grid_number,feature_number)))
        
        model.add(layers.Conv1D(40, 3, activation='relu', padding='same',kernel_initializer='he_normal'))
        model.add(layers.Conv1D(40, 3, activation='relu', padding='same',kernel_initializer='lecun_uniform'))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv1D(20, 3, activation='relu', padding='same',kernel_initializer='lecun_uniform'))
        model.add(layers.Conv1D(20, 3, activation='relu', padding='same',kernel_initializer='lecun_uniform'))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(400,activation='relu'))
        model.add(layers.Dense(100,activation='relu'))
        model.add(layers.Dense(10,activation='relu'))
        
        model.add(layers.Dense(1,activation="linear"))
        model.compile(optimizer=Adam(learning_rate=1e-5),loss='mse', metrics=[pearson_r])
        
        history = model.fit(train_feature, train_label, epochs=2000,batch_size=16,shuffle=True,verbose=0)
        saved_hist.append(history)
        a_predict = model.predict(test_feature)
        
        
        '''
        aa = []
        ss = a_predict.shape
        for f in range(ss[0]):
            aa.append(a_predict[f][0])
            #res.append(a_predict[f][0])
        bb = np.array(aa)
        #print(bb.shape,test_label.shape)
        #print(a_predict)
        pcc = scipy.stats.pearsonr(test_label,bb)
        print('************************************')
        print('test: ',pcc[0])
        final_pcc.append(pcc[0])
        #print(res)
        
        
        b_predict = model.predict(train_feature)
        aa = []
        ss = b_predict.shape
        for f in range(ss[0]):
            aa.append(b_predict[f][0])
        bb = np.array(aa)
        #print(bb.shape,train_label.shape)
        #print(a_predict)
        pcc = scipy.stats.pearsonr(train_label,bb)
        print('train: ',pcc)
        '''
        
        #model_name = 'cnn.h5'
        #model.save(model_name)
        
        middle = Model(inputs=model.input,outputs=model.layers[6].output)
        temp_res = middle.predict(test_feature)
        
        middle2 = Model(inputs=model.input,outputs=model.layers[9].output)
        temp_res2 = middle2.predict(test_feature)
        
        In = index[i]
        for kk in range(test_length):
            real_index = In[kk]
            res1[real_index,:] = temp_res[kk,:]
            res2[real_index,:] = temp_res2[kk,:]
            res3[real_index][0] = a_predict[kk][0]
        
        
    
    #filename = './skempi/cnn-feature/' + str(N) + '_40_20_feature800_' + typ + '_0.25.npy'
    #np.save(filename,res1)
    
    #filename = './skempi/cnn-feature/' + str(N) + '_40_20_feature10_' + typ + '_0.25.npy'
    #np.save(filename,res2)
    
    filename = './skempi/cnn-feature/' + str(N) + '_40_20_prediction_' + typ + '_0.25.npy'
    np.save(filename,res3)
    #print(typ,np.mean(final_pcc))

def get_combine(N):
    filename = './skempi/cnn-feature/' + N + '_40_20_prediction_mutate2_C_2_h0_0.25.npy'
    d1 = np.load(filename)
    for typ1 in site_typ:
        for typ2 in atom_typ:
            for typ3 in tor_typ:
                if typ1=='mutate2' and typ2=='C' and typ3=='2_h0':
                    continue
                name = site_typ[i] + '_' + atom_typ[j] + '_' + tor_typ[k]
                filename = './skempi/cnn-feature/' + N + '_40_20_prediction_' + name + '_0.25.npy'
                d2 = np.load(filename)
                d1 = np.hstack((d1,d2))
    filename = './skempi/cnn-feature/' + N + 'cnn.npy'
    np.save(filename,d1)

atom_typ = [ 'C','N','O','CN','CO','NO' ]
site_typ = [ 'mutate2','wild2' ]
tor_typ = [ '2_h0','3_h0','99_h0','99_h1','100_h0','100_h1' ]
for N in range(10):
    for i in range(2):
        for j in range(6):
            for k in range(6):
                name = site_typ[i] + '_' + atom_typ[j] + '_' + tor_typ[k]
                main(name,str(N))
                #print(name,typ,'ok')
    get_combine(str(N))

























