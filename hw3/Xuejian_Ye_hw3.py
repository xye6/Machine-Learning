#usr/bin/python3
# coding: utf-8

# In[1]:

import math
#import matplotlib.pyplot as plt
import numpy as np
import random

def data_format(data):   
    lines = data.readlines()
    data_list = [word.strip() for word in lines]

    new_data_list=[]
    for i in data_list:
        each_line=[]
        for j in range(len(i.split())):
            single_data= i.split()[j].split(':')[0]
            each_line.append(eval(single_data))
        new_data_list.append(each_line)

    labels=[]
    for i in range(len(new_data_list)):
        labels.append(new_data_list[i][0])

    for i in range(len(new_data_list)):
        new_data_list[i].pop(0)

    new_data_list_combine=[]
    for i in range(len(new_data_list)):
        x=[0]*123
        for j in new_data_list[i]:
            x[j-1]=1   
 #       x.append(1)   
        new_data_list_combine.append(x)

    X= np.array(new_data_list_combine)
    Y= np.array(labels)  
    return X,Y


# In[2]:

data1 = open("/u/cs246/data/adult/a7a.train")
X1,Y1=data_format(data1)

data2 = open("/u/cs246/data/adult/a7a.test")
X2,Y2=data_format(data2)

data3 = open("/u/cs246/data/adult/a7a.dev")
X3,Y3=data_format(data3)


# In[3]:

def svm_sgd(X,Y,n,C):
    weight = np.zeros(len(X[0]))
    b = 0
    lr = 0.1
    epochs = n
    for N in range(1,epochs+1):
        for i, x in enumerate(X):
            if (Y[i] * (np.dot(weight.T, X[i]) + b)) < 1:
                weight = weight - lr* ((1/len(X))*weight-C*(X[i]*Y[i]))
                b = b + lr*(C*Y[i])
            else:
                weight = weight-lr*(1/len(X))*weight
    weight=weight.tolist()
    weight.insert(0,b)
    
    return weight

#w = svm_sgd(X1,Y1,1,0.868)


# In[4]:

def svm_train(X,Y,w):
    b = w[0]
    weight = np.asarray(w[1:])
    lr = 0.1
    epochs = 1
    for N in range(1,epochs+1):
        err_count=0
        
        for i, x in enumerate(X):
            if (Y[i] * (np.dot(weight.T, X[i]) + b)) < 0:
                err_count += 1 
        accurate_rate = 1-err_count/len(X) 
#        print('TRAINING_ACCURACY:',accurate_rate) 

    return accurate_rate


# In[5]:

def svm_test(X,Y,w):
    b = w[0]
    weight = np.asarray(w[1:])
    lr = 0.1
    epochs = 1
    for N in range(1,epochs+1):
        err_count=0
        
        for i, x in enumerate(X):
            if (Y[i] * (np.dot(weight.T, X[i]) + b)) < 0:
                err_count += 1 
        accurate_rate = 1-err_count/len(X) 
#        print('TEST_ACCURACY:',accurate_rate) 

    return accurate_rate


# In[6]:

def svm_dev(X,Y,w):
    b = w[0]
    weight = np.asarray(w[1:])
    lr = 0.1
    epochs = 1
    for N in range(1,epochs+1):
        err_count=0
        
        for i, x in enumerate(X):
            if (Y[i] * (np.dot(weight.T, X[i]) + b)) < 0:
                err_count += 1 
        accurate_rate = 1-err_count/len(X) 
#        print('DEV_ACCURACY:',accurate_rate) 

    return accurate_rate


# In[7]:

def result(n,C):   
    print('EPOCHS: ',n)
    print('CAPACITY: ',C)

    w = svm_sgd(X1,Y1,n,C)
    print('TRAINING_ACCURACY:',svm_train(X1,Y1,w))
    print('TEST_ACCURACY:',svm_test(X2,Y2,w))      
    print('DEV_ACCURACY:',svm_dev(X3,Y3,w))
    print('FINAL_SVM:',w)


# In[8]:

import sys
def main():
    if '--epochs' in sys.argv and '--capacity' in sys.argv:
        ep = int(sys.argv[sys.argv.index('--epochs')+1])
        C = float(sys.argv[sys.argv.index('--capacity')+1])
        result(ep,C)
if __name__ == "__main__":
    main()
        


# In[ ]:



