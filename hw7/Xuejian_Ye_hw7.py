
# coding: utf-8

# In[1]:

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pylab
from scipy import stats
get_ipython().magic('matplotlib inline')


# In[2]:

#data
file_object = open("points.dat")
lines = file_object.readlines()
data_list = [word.strip() for word in lines]

new_data_list=[]
for i in data_list:
    each_line=[]
    for j in range(len(i.split())):
        single_data= i.split()[j].split(':')[0]
        each_line.append(eval(single_data))
    new_data_list.append(each_line)
    
new_data_list=mat(new_data_list)
devsample=new_data_list[900:,:]
trainsample=new_data_list[0:900,:]


# In[3]:

def initial_parameters(trainsample,K):   
    [N,D]=shape(trainsample)
    mean_sample=trainsample.mean(axis=0)
    min_sample=trainsample.min(axis=0)
    max_sample=trainsample.max(axis=0)
    cent=zeros((K,D))
    
    np.random.seed(5)
    random_data = np.random.permutation(trainsample)
    miu = random_data[:K]
    
    """
    for i in range(K):
        for j in range(D):
            cent[i,j]=mean_sample[0,j] + min_sample[0,j] + \
            (max_sample[0,j]-min_sample[0,j])*random.random()             
    miu=cent
    """ 
    sigma=zeros((K,D,D))
    for i in range(K):
        sigma[i,:,:]=cov(transpose(trainsample))
    pi=[1/K]*K
    
    return miu,pi,sigma 

def probability_of_eachpoint(trainsample,miu,sigma,pi,K):
    [N,D]=shape(trainsample)
    prob=zeros((len(trainsample),K))
    
    for i in range(K):
        t1=trainsample-miu[i,:] #X-miu
        inv_sigma = linalg.pinv(mat(sigma[i,:,:]))
        t2 = mat(sum(array((t1*inv_sigma))*array(t1),1)).T
        sigma_det = linalg.det(mat(inv_sigma))
        p=1/(2*(math.pi)) * sqrt(sigma_det)*exp(-0.5*t2)
        prob[:,i] = list(p[:,0])
        
    return prob


# In[4]:

def EM_step(trainsample,devsample,K,cov_type):
    type_of_cov = cov_type
    [N, D] = shape(trainsample)
    [miu,pi,sigma]=initial_parameters(trainsample,K)
    log_likelihood=[]
    dev_log_likelihood = []

    for i in range(80): 
        
        prob=probability_of_eachpoint(trainsample,miu,sigma,pi,K)           
        pgamma = mat(array(prob) * array(tile(pi,(N,1))))
        pgamma = pgamma / tile(sum(pgamma,1),(1,K))
        
        Nk=sum(pgamma,0)
        miu=mat(diag((1/Nk).tolist()[0])) * (pgamma.T)* trainsample
        pi=Nk/N

        for j in range(K):
            Xshift = trainsample - tile(miu[j],(N,1))
            t=(Xshift.T * mat(diag(pgamma[:,j].T.tolist()[0]))*Xshift)/Nk[:,j]
            sigma[j,:,:]=t

        if type_of_cov == 'Tied':
            a1=sum(sigma[:,0,0])/K
            a2=sum(sigma[:,0,1])/K
            a3=sum(sigma[:,1,0])/K
            a4=sum(sigma[:,1,1])/K
            b = zeros((K,D,D))
            c = [[a1,a2],[a3,a4]]
            for i in range(K):
                b[i,:,:] = c              
            sigma = b
               
        L1 = sum(log(prob*(mat(pi).T)))
        log_likelihood.append(L1)
        
        prob_dev=probability_of_eachpoint(devsample,miu,sigma,pi,K)
        L2 = sum(log(prob_dev*(mat(pi).T)))
        dev_log_likelihood.append(L2)
        
    return log_likelihood, dev_log_likelihood


# In[5]:

import numpy as np
x= []
for i in range(79):
    x.append(i)

fig,ax=plt.subplots(2,2,figsize=(15,12))
iteration = 80
_range = range(1,iteration+1,1)

for group in range(5):
    [se_train,se_dev]=EM_step(trainsample,devsample,group+2,'separate')
    [Tied_train,Tied_dev]=EM_step(trainsample,devsample,group+2,'Tied')
    
    ax[0][0].plot(x, se_train[1:], label="{} clusters".format(group+2))
    ax[0][1].plot(x, se_dev[1:], label="{} clusters".format(group+2))
    ax[1][0].plot(x, Tied_train[1:], label="{} clusters".format(group+2))
    ax[1][1].plot(x, Tied_dev[1:], label="{} clusters".format(group+2))
    
    
for i in range(2):
    for j in range(2):
        ax[i][j].set_xlim([1,iteration+1])
        ax[i][j].set_xlabel('iteration')
        ax[i][j].set_ylabel('log likelihood')
        ax[i][j].legend(loc = 'lower right',fontsize='small')
        
ax[0][0].set_title('trainsample with seperate covariance')
ax[0][1].set_title('devsample with seperate covariance')
ax[1][0].set_title('trainsample with tied covariance')
ax[1][1].set_title('devsample with tied covariance')


# In[ ]:




# In[ ]:




# In[ ]:



