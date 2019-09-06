# -*- coding: utf-8 -*-
from numpy import *
from pylab import *
import random as random
import matplotlib.pyplot as plt  
import math
trainsample=[]




def calprop(trainsample,miu,sigma,pi,K):
    gamma=zeros((len(trainsample),K))
    prob=zeros((len(trainsample),K))
    for k in range(K):
        temp1=trainsample[:,:]-miu[k,:]
        invsigma = linalg.pinv(mat(sigma[k, :, :]))
        temp2 = mat(sum(array((temp1*invsigma)) * array(temp1), 1)).T
        detsigma = linalg.det(mat(invsigma))
        if detsigma <= 0:
            detsigma=0.0000001
        p=1/(2*(math.pi)) * sqrt(detsigma)*exp(-0.5*temp2)
        prob[:,k] = list(p[:,0])
    return prob

def calgamma(trainsample,miu,sigma,pi,K):
    prob=calprop(trainsample,miu,sigma,pi,K)
    gamma=zeros((shape(prob)))
        #print gamma
    sumk=[]
    for i in range(len(trainsample)):
        sumki=0
        for k in range(K):
            sumki=sumki+pi[k]*prob[i,k]
        sumk.append(sumki)
    for i in range(len(trainsample)):
        for k in range(K):
            gamma[i,k]=pi[k]*prob[i,k]/sumk[i]
    return gamma
def initmiu(K,trainsample):
    row,col=shape(trainsample)
    meana=trainsample.mean(axis=0)
    #print(meana)
    mina=trainsample.min(axis=0)
    maxa=trainsample.max(axis=0)
    cent=zeros((K,col))
    for i in range(K):
        for j in range(col):
            cent[i,j]=mina[0,j]+(maxa[0,j]-mina[0,j])*random.random()
            #cent[i,j]=-0.25+0.3*random.random()
    return cent

def initall(trainsample,cent,K):
    num,dim=shape(trainsample)
    miu=cent

    pi=zeros((1,K))
    sigma=zeros((K,dim,dim))
    dis=zeros((num,K))
    for i in range(num):
        for j in range(K):
            dif=trainsample[i,:]-cent[j,:]
            dis[i,j]=multiply(dif,dif).sum()
    
    labels=dis.argmin(axis=1)
    for i in range(K):
        labellist=[]
        for ii in range(len(labels)):
            if labels[ii]==i:
                labellist.append(ii)
        #print(labellist)
        traink=trainsample[labellist,:]
        pi[0,i]=shape(traink)[0]/double(num)
        #print(cov(transpose(traink)))
        
        sigma[i,:,:]=cov(transpose(traink))
        sigma[i,:,:]=nan_to_num(sigma[i,:,:])
    pi=list(pi[0,:])
    return miu,pi,sigma

fp= open('points.dat')
for line in fp.readlines(): 
    tempdata=line.split(" ")
    temp=[]
    for i in range(len(tempdata)):
        if tempdata[i]!="":
          if tempdata[i][-1]=="\n":
              temp.append(double(tempdata[i][:-1]))
          else:
              temp.append(double(tempdata[i]))
    trainsample.append(temp)
trainsample=mat(trainsample)
valsample=trainsample[900:,:]
trainsample=trainsample[0:900,:]

K=6
tiedmode=False

cent=initmiu(K,trainsample)
miu,pi,sigma=initall(trainsample,cent,K)

threshold=0.1
Lp=-100000
num,dim=shape(trainsample)
iter=0
trainlike=[]
vallike=[]
iterlist=[]
while(True):

    prob=calprop(trainsample,miu,sigma,pi,K)
    gamma=zeros((shape(prob)))

    sumk=[]
    for i in range(len(trainsample)):
        sumki=0
        for k in range(K):
            sumki=sumki+pi[k]*prob[i,k]
        sumk.append(sumki)
    for i in range(len(trainsample)):
        for k in range(K):
            gamma[i,k]=pi[k]*prob[i,k]/sumk[i]    

    n1=gamma.sum(axis=0)
    n=mat(n1)
    sigma=zeros((K,dim,dim))
    miu=divide(dot(gamma.T,trainsample),n.T)
    pi=divide(n1,num)
    
    for i in range(num):
      for k in range(K):
        temp=trainsample[i,:]-miu[k,:]
        temp2=multiply(gamma[i,k],temp)

        sigma[k,:,:]=sigma[k,:,:]+divide(dot(temp2.T,temp),n[0,k])
    if tiedmode==True:
        aaa=zeros((2,2))
        for k in range(K):
            aaa=aaa+sigma[k,:,:]
        aaa=aaa/K
        for k in range(K):
            sigma[k,:,:]=aaa

    
    L=log(dot(prob,pi)).sum()
    print("train iter= "+str(iter)+", loglikelihood for train set is "+str(L))
    if L-Lp<threshold:
        break
    Lp=L
    probval=calprop(valsample,miu,sigma,pi,K) 
    Lval=log(dot(probval,pi)).sum()
    print("loglikelihood for validation set is "+str(Lval))
    iterlist.append(iter)   
    iter=iter+1
    vallike.append(Lval)
    trainlike.append(L)
# probval=calprop(valsample,miu,sigma,pi,K)       
# Lval=log(dot(probval,pi)).sum()
print("K = "+str(K)+" ,final centroids are:")
print(miu)

fig, ax = plt.subplots(2)
plt1=ax[0]
plt2=ax[1]
# plt1.ylim(min(trainlike),max(trainlike)+100)
# plt1.xlim(0,len(iterlist))
plt1.set_ylabel("log-likelihood of train set",size=18)
plt1.set_xlabel("iter number",size=18)

plt1.plot(trainlike,'ro-',linewidth=2)# use pylab to plot x and y
#plt.plot(y, x,'*')

plt2.set_ylabel("log-likelihood of validation set",size=18)
plt2.set_xlabel("iter number",size=18)
# plt2.ylim(min(vallike),max(vallike)+100)
# plt2.xlim(0,len(iterlist))
#plt2.xlabel("iter number",size=18)
#plt2.ylabel("log-likelihood of validation set",size=18)

plt2.plot(vallike,'ro-',linewidth=2)# use pylab to plot x and y
#plt.plot(y, x,'*')
plt.show()# show t
