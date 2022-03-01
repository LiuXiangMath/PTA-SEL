# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import gudhi

Pre = './s645/'

def get_info(mut):
    chainid = mut[0]
    wildtype = mut[2]
    mutanttype = mut[-1]
    residueid = mut[3:-1]
    return chainid, wildtype, mutanttype, residueid

def distance_of_two_points(p1,p2):
    temp = pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2) + pow(p1[2]-p2[2],2)
    res = pow(temp,0.5)
    return res


def get_bar_from_point(point,dim):
    rips_complex = gudhi.RipsComplex(points=point,max_edge_length=10)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim+1)
    diag = simplex_tree.persistence()
    res0 = []
    res1 = []
    res2 = []
    delete_number = 0
    for i in range(len(diag)):
        item = diag[i]
        if item[0]==0:
            if item[1][1]!=float('inf'):
                res0.append( [ item[1][0],item[1][1] ] )
            elif item[1][1]==float('inf') and delete_number==0:
                delete_number = delete_number + 1
            else:
                res0.append( [ item[1][0],item[1][1] ] )
        elif item[0]==1:
            res1.append( [ item[1][0],item[1][1] ] )
        elif item[0]==2:
            res2.append( [ item[1][0],item[1][1] ] )
    if dim==1:
        return res0,res1
    elif dim==2:
        return res0,res1,res2
    
    
    
def get_max(ls):
    if len(ls)==0:
        return 0
    return max(ls)
    

def get_min(ls):
    if len(ls)==0:
         return 0
    return min(ls)
    

def get_ave(ls):
    if len(ls)==0:
        return 0
    return np.mean(ls)
    

def get_std(ls):
    if len(ls)==0:
        return 0
    return np.std(ls)
    

def get_sum(ls):
    if len(ls)==0:
        return 0
    return sum(ls)


def get_statistic_feature_from_bar_5(bar,dim):
    # step:0.5
    # endpoint: max, min, average, number, sum, std
    value = []
    for item in bar:
        if dim==1:
            value.append(item[0])
        value.append(item[1])
    
    res = np.zeros((1,120))
    t = 20
    for i in range(20):
        left = i * 0.5
        right = i * 0.5 + 0.5
        value_temp = []
        for item in value:
            if item>=left and item<=right:
                value_temp.append(item)
        
        res[0][i*6] = get_max(value_temp)
        res[0][i*6+1] = get_min(value_temp)
        res[0][i*6+2] = get_ave(value_temp)
        res[0][i*6+3] = len(value_temp)
        res[0][i*6+4] = get_sum(value_temp)
        res[0][i*6+5] = get_std(value_temp)
        
    return res



def get_statistic_feature_from_bar_25(bar,dim):
    # step:0.25
    # endpoint: max, min, average, number, sum, std
    value = []
    for item in bar:
        if dim==1:
            value.append(item[0])
        value.append(item[1])
    
    res = np.zeros((1,240))
    t = 40
    for i in range(40):
        left = i * 0.25
        right = i * 0.25 + 0.25
        value_temp = []
        for item in value:
            if item>=left and item<=right:
                value_temp.append(item)
        
        res[0][i*6] = get_max(value_temp)
        res[0][i*6+1] = get_min(value_temp)
        res[0][i*6+2] = get_ave(value_temp)
        res[0][i*6+3] = len(value_temp)
        res[0][i*6+4] = get_sum(value_temp)
        res[0][i*6+5] = get_std(value_temp)
        
    return res


def get_tor_algebra_I_J(point,J,outfile,typ):
    N1 = 120
    N2 = 240
    t1 = point.shape
    res0 = []
    res1 = []
    res2 = []
    # point number <=1
    if len(t1)==1:
        res_tor = np.zeros((1,N1))
        np.save(outfile+'h0_statistic_0.5.npy',res_tor)
        np.save(outfile+'h1_statistic_0.5.npy',res_tor)
        res_tor = np.zeros((1,N2))
        np.save(outfile+'h0_statistic_0.25.npy',res_tor)
        np.save(outfile+'h1_statistic_0.25.npy',res_tor)
        
        return
    # J==2, betti0
    if J==2:
        for k1 in range(t1[0]):
            for k2 in range(k1+1,t1[0]):
                dis = distance_of_two_points(point[k1],point[k2])
                res0.append( [0,dis] )
        res_tor = get_statistic_feature_from_bar_25(res0, 0)
        np.save(outfile+'h0_statistic_0.25.npy',res_tor)
        
        res_tor0 = get_statistic_feature_from_bar_5(res0,0)
        np.save(outfile+'h0_statistic_0.5.npy',res_tor0)
        return
    
    # J==3, betti0
    if J==3:
        if t1[0]<=2:
            res_tor = np.zeros((1,N1))
            np.save(outfile+'h0_statistic_0.5.npy',res_tor)
            res_tor = np.zeros((1,N2))
            np.save(outfile+'h0_statistic_0.25.npy',res_tor)
            
            return
        else:
            for k1 in range(t1[0]):
                for k2 in range(k1+1,t1[0]):
                    dis1 = distance_of_two_points(point[k1],point[k2])
                    for k3 in range(k2+1,t1[0]):
                        dis2 = distance_of_two_points(point[k1],point[k3])
                        dis3 = distance_of_two_points(point[k2],point[k3])
                        m = min(dis1,dis2,dis3)
                        M = max(dis1,dis2,dis3)
                        res0.append( [0,m] )
                        res0.append( [0,dis1+dis2+dis3-m-M] )
            res_tor0 = get_statistic_feature_from_bar_5(res0,0)
            np.save(outfile+'h0_statistic_0.5.npy',res_tor0)
            res_tor0 = get_statistic_feature_from_bar_25(res0,0)
            np.save(outfile+'h0_statistic_0.25.npy',res_tor0)
    
            return
    '''
    # J==4, betti0 and betti1
    if J==4:
        if t1[0]<=3:
            res_tor = np.zeros((1,N))
            np.save(outfile+'h0.npy',res_tor)
            np.save(outfile+'h1.npy',res_tor)
            return
        else:
            for k1 in range(t1[0]):
                for k2 in range(k1+1,t1[0]):
                    for k3 in range(k2+1,t1[0]):
                        for k4 in range(k3+1,t1[0]):
                            temp1_point = [ 
                                [point[k1][0],point[k1][1],point[k1][2]],
                                [point[k2][0],point[k2][1],point[k2][2]],
                                [point[k3][0],point[k3][1],point[k3][2]],
                                [point[k4][0],point[k4][1],point[k4][2]],
                                           ]
                            temp_zero_bar,temp_one_bar = get_bar_from_point(temp1_point,1)
                            for item0 in temp_zero_bar:
                                res0.append(item0)
                            for item1 in temp_one_bar:
                                res1.append(item1)
                       
            res_tor0 = get_feature_from_bar(res0)
            np.save(outfile+'h0.npy',res_tor0)
            res_tor1 = get_feature_from_bar(res1)
            np.save(outfile+'h1.npy',res_tor1)
            return
        
    # J==5, betti0, betti1 and betti2
    if J==5:
        if t1[0]<=4:
            res_tor = np.zeros((1,N))
            np.save(outfile+'h0.npy',res_tor)
            np.save(outfile+'h1.npy',res_tor)
            np.save(outfile+'h2.npy',res_tor)
            return
        else:
            for k1 in range(t1[0]):
                for k2 in range(k1+1,t1[0]):
                    for k3 in range(k2+1,t1[0]):
                        for k4 in range(k3+1,t1[0]):
                            for k5 in range(k4+1,t1[0]):
                                temp1_point = [ 
                                    [point[k1][0],point[k1][1],point[k1][2]],
                                    [point[k2][0],point[k2][1],point[k2][2]],
                                    [point[k3][0],point[k3][1],point[k3][2]],
                                    [point[k4][0],point[k4][1],point[k4][2]],
                                    [point[k5][0],point[k5][1],point[k5][2]]
                                               ]
                                temp_zero_bar,temp_one_bar,temp_two_bar = get_bar_from_point(temp1_point,2)
                                for item0 in temp_zero_bar:
                                    res0.append(item0)
                                for item1 in temp_one_bar:
                                    res1.append(item1)
                                for item2 in temp_two_bar:
                                    res2.append(item2)
                       
            res_tor0 = get_feature_from_bar(res0)
            np.save(outfile+'h0.npy',res_tor0)
            res_tor1 = get_feature_from_bar(res1)
            np.save(outfile+'h1.npy',res_tor1)
            res_tor2 = get_feature_from_bar(res2)
            np.save(outfile+'h2.npy',res_tor2)
            
            return
            
    '''
    # J==n, betti0, betti1
    if J==100:
        if typ=='alpha':
             alpha_complex = gudhi.AlphaComplex(points=point)
             simplex_tree = alpha_complex.create_simplex_tree()
        else:
        
            rips_complex = gudhi.RipsComplex(points=point,max_edge_length=10)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        diag = simplex_tree.persistence()
        res0 = []
        res1 = []
        delete_number = 0
        for i in range(len(diag)):
            item = diag[i]
            if item[0]==0:
                if item[1][1]!=float('inf'):
                    res0.append( [ item[1][0],item[1][1] ] )
                elif item[1][1]==float('inf') and delete_number==0:
                    delete_number = delete_number + 1
                else:
                    res0.append( [ item[1][0],item[1][1] ] )
            elif item[0]==1:
                res1.append( [ item[1][0],item[1][1] ] )
                       
        res_tor0 = get_statistic_feature_from_bar_5(res0,0)
        np.save(outfile+'h0_statistic_0.5.npy',res_tor0)
        res_tor1 = get_statistic_feature_from_bar_5(res1,1)
        np.save(outfile+'h1_statistic_0.5.npy',res_tor1)
            
        res_tor0 = get_statistic_feature_from_bar_25(res0,0)
        np.save(outfile+'h0_statistic_0.25.npy',res_tor0)
        res_tor1 = get_statistic_feature_from_bar_25(res1,1)
        np.save(outfile+'h1_statistic_0.25.npy',res_tor1)
        return
    
    # J==n-1, betti0, betti1
    if J==99:
        for k1 in range(t1[0]):
            new_point = np.zeros((t1[0]-1,3))
            c = 0
            for k2 in range(t1[0]):
                if k2!=k1:
                    new_point[c,:] = point[k2,:]
                    c = c + 1
            temp_zero_bar,temp_one_bar = get_bar_from_point(new_point,1)
            for item0 in temp_zero_bar:
                res0.append(item0)
            for item1 in temp_one_bar:
                res1.append(item1)
        res_tor0 = get_statistic_feature_from_bar_5(res0,0)
        np.save(outfile+'h0_statistic_0.5.npy',res_tor0)
        res_tor1 = get_statistic_feature_from_bar_5(res1,1)
        np.save(outfile+'h1_statistic_0.5.npy',res_tor1)
            
        res_tor0 = get_statistic_feature_from_bar_25(res0,0)
        np.save(outfile+'h0_statistic_0.25.npy',res_tor0)
        res_tor1 = get_statistic_feature_from_bar_25(res1,1)
        np.save(outfile+'h1_statistic_0.25.npy',res_tor1)
        return
    
    # J==n-2, betti0, betti1
    if J==98:
        if t1[0]<=2:
            res_tor = np.zeros((1,N1))
            np.save(outfile+'h0_statistic_0.5.npy',res_tor)
            np.save(outfile+'h1_statistic_0.5.npy',res_tor)
            res_tor = np.zeros((1,N2))
            np.save(outfile+'h0_statistic_0.25.npy',res_tor)
            np.save(outfile+'h1_statistic_0.25.npy',res_tor)
            return
        
        for k1 in range(t1[0]):
            for k2 in range(k1+1,t1[0]):
                new_point = np.zeros((t1[0]-2,3))
                c = 0
                for k3 in range(t1[0]):
                    if k3!=k1 and k3!=k2:
                        new_point[c,:] = point[k3,:]
                        c = c + 1
                temp_zero_bar,temp_one_bar = get_bar_from_point(new_point,1)
                for item0 in temp_zero_bar:
                    res0.append(item0)
                for item1 in temp_one_bar:
                    res1.append(item1)
        res_tor0 = get_statistic_feature_from_bar_5(res0,0)
        np.save(outfile+'h0_statistic_0.5.npy',res_tor0)
        res_tor1 = get_statistic_feature_from_bar_5(res1,1)
        np.save(outfile+'h1_statistic_0.5.npy',res_tor1)
            
        res_tor0 = get_statistic_feature_from_bar_25(res0,0)
        np.save(outfile+'h0_statistic_0.25.npy',res_tor0)
        res_tor1 = get_statistic_feature_from_bar_25(res1,1)
        np.save(outfile+'h1_statistic_0.25.npy',res_tor1)
        return
    
    
def Tor_algebra_to_file3(start,end,filtration):
    filename = Pre + 'AB-Bind_S645.xlsx'
    df1 = pd.read_excel(filename)
    t1 = df1.shape
    for i in range(start,end):
        #print(i)
        pdbid = df1.iloc[i,0]
        chainid, wildtype, mutanttype, residueid = get_info(df1.iloc[i,1])
        folder = pdbid + '_' + chainid + '_' + wildtype + '_' + residueid + '_' + mutanttype
        for typ in ['mutation_mut', 'mutation_wild']:
            for atom in ['C','N','O','CN','CO','NO']:
                filename1 = Pre + 's645_point_10/' + folder + '_' + typ + '_' + atom + '1.txt'
                filename2 = Pre + 's645_point_10/' + folder + '_' + typ + '_' + atom + '2.txt'
                point1 = np.loadtxt(filename1,delimiter=',')
                point2 = np.loadtxt(filename2,delimiter=',')
                #print(typ,point1.shape,point2.shape)
                
                
                for J in [ 2, 3, 99, 100 ]:
                    filepath1 =Pre + 's645_tor_algebra_' + str(filtration) + '/' + folder + '_' + typ + '_' + atom + '1_tor_' + str(J) + '_' 
                    get_tor_algebra_3(point1,filepath1)
                
                    filepath2 =Pre + 's645_tor_algebra_' + str(filtration) + '/' + folder + '_' + typ + '_' + atom + '2_tor_' + str(J) + '_' 
                    get_tor_algebra_3(point2,filepath2)
                    
                
        print(i,folder)
        
        
        
def Tor_algebra_each_feature_to_file(start,end,filtration,grid_size,typ):
    filename = Pre + 'AB-Bind_S645.xlsx'
    df1 = pd.read_excel(filename)
    t1 = df1.shape
    row = 645
    N = 1
    number = 100
    grid_number = int(filtration/grid_size)
    grid_number = number
    column = 2 * 2 * 6 * 240
    feature_matrix = np.zeros((row,column))
    
    J = typ
    for i in range(start,end):
        #print(i)
        count = 0
        pdbid = df1.iloc[i,0]
        chainid, wildtype, mutanttype, residueid = get_info(df1.iloc[i,1])
        folder = pdbid + '_' + chainid + '_' + wildtype + '_' + residueid + '_' + mutanttype
        
        print(i,folder)
        for atom in ['C','N','O','CN','CO','NO']:
            #for J in ['2_h0', '3_h0', '4_h0', '4_h1', '98_h0', '98_h1', '99_h0', '99_h1', '100_h0', '100_h1']:
            #for J in ['2_h0', '3_h0', '4_h0', '4_h1', '98_h0', '99_h0', '100_h0']:
            
                    # mutation_mut
                    filename5 = Pre + 'rips_s645_tor_algebra_' + str(filtration) + '/' + folder \
                        + '_mutation_mut_' + atom + '1_tor_' + J + '.npy'
                    filename6 = Pre + 'rips_s645_tor_algebra_' + str(filtration) + '/' + folder \
                        + '_mutation_mut_' + atom + '2_tor_' + J + '.npy'
                    
                   
                    # mutation_wild
                    filename7 = Pre + 'rips_s645_tor_algebra_' + str(filtration) + '/' + folder \
                        + '_mutation_wild_' + atom + '1_tor_' + J + '.npy'
                    filename8 = Pre + 'rips_s645_tor_algebra_' + str(filtration) + '/' + folder \
                        + '_mutation_wild_' + atom + '2_tor_' + J + '.npy'
                        
                    
                    bar5 = np.load(filename5)
                    bar6 = np.load(filename6)
                    bar7 = np.load(filename7)
                    bar8 = np.load(filename8)
                    
                
                    
                    for j1 in range(6):
                        for j2 in range(N):
                            feature_matrix[i,count] = bar5[0,j2*6+j1]
                            count = count + 1
                    
                    for j1 in range(6):
                        for j2 in range(N):
                            feature_matrix[i,count] = bar6[0,j2*6+j1]
                            count = count + 1
                    
                    for j1 in range(6):
                        for j2 in range(N):
                            feature_matrix[i,count] = bar7[0,j2*6+j1]
                            count = count + 1
                    
                    for j1 in range(6):
                        for j2 in range(N):
                            feature_matrix[i,count] = bar8[0,j2*6+j1]
                            count = count + 1
                    
                    
                    
    filename = Pre + 'feature/' + typ + '_statistic_feature_0.25.npy'
    np.save(filename,feature_matrix)
          

def seperate_feature(typ):
    filename = './s645/feature/' + typ + '_statistic_feature_0.25.npy'
    feature = np.load(filename)
    t1 = feature.shape
    atom_typ = ['C','N','O','CN','CO','NO']
    site_typ = [ 'mutate1', 'mutate2', 'wild1', 'wild2' ]
    
    
    for i in range(6):
        for j in range(4):
            res = np.zeros(( t1[0],240 ))
            left = i*4 + j
            right = i*4 + j + 1
            for k in range(t1[0]):
                res[k,:] = feature[k,left*240:right*240]
            filename = './skempi/feature/separate-feature/' + site_typ[j] + '_' + atom_typ[i] + '_' + typ + '.npy'
            np.save(filename,res)
    
   

Tor_algebra_to_file(0,645,10)
for typ in ['2_h0','3_h0','99_h0','99_h1','100_h0','100_h1']:
    Tor_algebra_each_feature_to_file(0,645,10,0.25,typ)
    seperate_feature(typ)
    
         
            