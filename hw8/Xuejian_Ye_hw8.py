
# coding: utf-8

# In[7]:

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pylab
from scipy import stats
import sys
get_ipython().magic('matplotlib inline')

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


# In[8]:

def initial_parameters(trainsample,K):
    A=zeros((K,K), dtype=np.longdouble)
    np.random.seed(3)
    for i in range(K):
        t=np.random.rand(K)
        A[i]=t/sum(t)
        
    pi=[1/K]*K

    [N,D]=shape(trainsample)
    np.random.seed(5)
    random_data = np.random.permutation(trainsample)
    miu = random_data[:K]
    sigma=zeros((K,D,D))
    for i in range(K):
        sigma[i,:,:]=cov(transpose(trainsample))
        
    [N,D]=shape(trainsample)
    prob=zeros((len(trainsample),K))
    for i in range(K):
        t1=trainsample-miu[i,:] #X-miu
        inv_sigma = linalg.pinv(mat(sigma[i,:,:]))
        t2 = mat(sum(array((t1*inv_sigma))*array(t1),1)).T
        sigma_det = linalg.det(mat(inv_sigma))
        p=1/(2*(math.pi)) * sqrt(sigma_det)*exp(-0.5*t2)
        prob[:,i] = list(p[:,0])
    
    B=sum(prob,1)
    
    return A,B,pi,miu,sigma,prob,p


# In[9]:

def alpha_beta(trainsample,K,A,prob,pi):   
    [N,D]=shape(trainsample)
    alpha = np.zeros((N, K), dtype=np.longdouble)
    for i in range(N):
        if i == 0:
            alpha[i,:]=prob[i,:]*pi
        else:
            alpha[i,:]=alpha[i-1,].dot(A)*prob[i,:]
            
    beta = np.zeros((N, K), dtype=np.longdouble)
    for i in range(N-1,-1,-1):
        if i == N-1:
            beta[i,:]=np.array([1] * K)
        else:
            beta[i,:]=(prob[i+1,:] * beta[i+1,:]).dot(A.T)
    
    return alpha,beta


# In[10]:

def transition_matrix(K,N,alpha,beta,A,prob):
    
    A_n = np.zeros((K, K), dtype=np.longdouble)
    temp1= np.zeros((N,K), dtype=np.longdouble)
    for i in range(K):
        for n in range(N):
            if n!=N-1:
                temp1[n,:]=alpha[n,i]*beta[n+1,:]*A[i,:]*prob[n+1,:]/sum(alpha[N-1,:])
                A_n[i,:]=sum(temp1,0)
            else:
                temp1[n,:]=(alpha[n,1]*A[i,:])/np.sum(alpha[N-1,:])
                A_n[i,:]=sum(temp1,0)
    A_n = A_n/sum(A_n, axis=0)

    return A_n


# In[11]:

def EM_HMM(trainsample,devsample,K, cov_type= None):
    log_likelihood=[]
    dev_log_likelihood = []
    
    [N,D]=shape(trainsample)
    [A,B,pi,miu,sigma,prob,p]=initial_parameters(trainsample,K)
    [alpha,beta]=alpha_beta(trainsample,K,A,prob,pi)
#     print(alpha)
#     print(beta)
#    A_n=transition_matrix(K,alpha,beta,A)

    for i in range(40):
        #E-step
        rnk = (alpha * beta)/sum(alpha[N-1,:])
        temp1=array(mat(sum(rnk,0)).T)
        #M-step
        miu = array(trainsample.T.dot(rnk).T)/temp1
        sigma=zeros((K,D,D))
        for i in range(N):
            for k in range(K):
                sigma[k] = sigma[k] + rnk[i, k]*np.outer((trainsample[i]-miu[k]), (trainsample[i]-miu[k]))
        sigma = [sigma[i]/sum(rnk[:,i]) for i in range(K)]
#         for i in range(K):
#             sigma[i] =sigma[i]/sum(rnk[:,i])
#        print(sigma)
            
        if cov_type == 'tied':
            sigma = [sum(sigma)/K for i in range(K)]
                   
        pi=rnk[0,:]
        A = transition_matrix(K,N,alpha,beta,A,prob)
        [alpha,beta]=alpha_beta(trainsample,K,A,prob,pi)
        log_likelihood.append(np.log(sum((alpha[N-1,]))))
        
        [N2,D2]=shape(devsample)
        [alpha_dev,beta_dev]=alpha_beta(devsample,K,A,prob,pi)
        dev_log_likelihood.append(np.log(sum(alpha_dev[N2-1,])))
        
    return log_likelihood, dev_log_likelihood


# In[ ]:




# In[12]:

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
matplotlib.rcParams['figure.dpi'] = 150
set_matplotlib_formats('retina')

x=[]
for i in range(39):
    x.append(i)
    
fig,ax=plt.subplots(1,2,figsize=(12,5))
iteration=40
_range = range(1,iteration+1,1)


for K in range(2,8):
    like_train,like_dev=EM_HMM(trainsample,devsample,K)
    ax[0].plot(x,like_train[1:],label='{} clusters'.format(K))
    ax[1].plot(x,like_dev[1:],label='{} clusters'.format(K))
    
for i in range(2):
    ax[i].set_xlim([1,iteration+1])
    ax[i].set_xlabel('Iterations')
    ax[i].set_ylabel('Log likelihood')
    ax[i].legend(loc = 'lower right',fontsize='small')

    
ax[0].set_title('Train data')
ax[1].set_title('Development data') 


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



