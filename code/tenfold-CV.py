

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import scipy as sp


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
    



def get_combined_feature():
    filename1 = 'skempi/gbt-feature/0gbt.npy'
    d1 = np.load(filename1)
    filename2 = 'skempi/cnn-feature/0cnn.npy'
    d2 = np.loadtxt(filename2,delimiter=',')
    res = np.hstack((d1,d2))
    filename = 'skempi/stacking/0gbt_cnn.npy'
    np.save(filename,res)
    
    


def gradient_boosting(i,X_train,Y_train,X_test,Y_test):
    params={'n_estimators': 4000, 'max_depth': 6, 'min_samples_split': 2,
                'learning_rate': 0.01, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train,Y_train)
    a_predict = regr.predict(X_test)
    pearson_coorelation = sp.stats.pearsonr(Y_test,a_predict)
    mse1 = mean_squared_error(Y_test, regr.predict(X_test))
    mse2 = pow(mse1,0.5)
    #mse3 = mse2/0.7335
    mse3 = mse2
    return [pearson_coorelation[0],mse3]




def separate_train_and_test_index():
    pre = 'feature/'
    filename1 = pre + 'label.csv'
    label = np.loadtxt(filename1,delimiter=',')
    #print(label.shape)
    temp1 = []
    for i in range(1131):
        #print(label[i])
        temp1.append( [ i,label[i] ] )
    temp2 = sorted(temp1,key=lambda x:(x[1]))
    
    index_list = []
    for item in temp2:
        index_list.append(item[0])
    
    res = []
    for start in range(10):
        temp3 = []
        for j in range(start,1131,10):
            temp3.append( index_list[j] )
        res.append(temp3)
    print(res)
    
    filename2 = pre + '10crossvalidation_index.txt'
    f = open(filename2,'w')
    f.write(str(res))
    f.close()
    
    

 
def separate_train_and_test_index_nonbinder():
    pre = 'feature/'
    filename1 = pre + 'label.csv'
    label = np.loadtxt(filename1,delimiter=',')
    #print(label.shape)
    temp1 = []
    for i in range(645):
        #print(label[i])
        temp1.append( [ i,label[i] ] )
    temp2 = sorted(temp1,key=lambda x:(x[1]))
    
    index_list = []
    for item in temp2[:-27]:
        index_list.append(item[0])
    
    res = []
    for start in range(10):
        temp3 = []
        for j in range(start,645-27,10):
            temp3.append( index_list[j] )
        res.append(temp3)
    print(res)
    
    filename2 = pre + 'nonbinder_10crossvalidation_index.txt'
    f = open(filename2,'w')
    f.write(str(res))
    f.close()
    
    
    
def cross_validation():
    pre = './skempi/feature/'
    filename1 = pre + '10crossvalidation_index.txt'
    f = open(filename1)
    pre_index = f.read()
    index = eval(pre_index)
    f.close()
    all_number = 1131
    filename2 = pre + '0gbt_cnn.npy'
    feature_matrix = np.load(filename2)
    feature_matrix = normalize(feature_matrix)
    feature_shape = feature_matrix.shape
    print(feature_shape)
    filename3 = pre + 'label.csv'
    label = np.loadtxt(filename3,delimiter=',')
    ten_pcc = []
    ten_rmse = []
    for i in range(10):
        #print('CV',i)
        subindex = index[i]
        temp = len(subindex)
        test_feature = np.zeros((temp,feature_shape[1]))
        pre_test_label = []
        
        train_feature = np.zeros((all_number-temp,feature_shape[1]))
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
        #print(train_feature.shape,train_label.shape,test_feature.shape,test_label.shape)
        
        
        number = 10
        P = np.zeros((number,1))
        M = np.zeros((number,1))
        for k in range(10):
            [P[k][0],M[k][0]] = gradient_boosting(k,train_feature,train_label,test_feature,test_label)
            #print(P[k])
        
        median_p = np.median(P)
        median_m = np.median(M)
        ten_pcc.append(median_p)
        ten_rmse.append(median_m)
        print('CV ',i,median_p,median_m)
    print('PCC:',np.mean(ten_pcc))

    
    
cross_validation()