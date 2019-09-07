
# coding: utf-8
import numpy as np
import sys


file_object = open("/u/cs246/data/adult/a7a.train")
lines = file_object.readlines()
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
#    print(new_data_list[i])
    for j in new_data_list[i]:
        x[j-1]=1   
    x.append(1)   
    new_data_list_combine.append(x)

import numpy as np
X= np.array(new_data_list_combine)
Y= np.array(labels)

def perceptron_train(X,Y,n):
    weight=np.zeros(len(X[0]))  
    learning_rate=1
    epoches = n

    for t in range(epoches):
        err_count=0
        
        for i ,p in enumerate(X):
            if (np.dot(X[i], weight)*Y[i]) <= 0:
                err_count += 1
                weight = weight + learning_rate * X[i] * Y[i]
                
    accurate_rate = 1-err_count/len(X)
       
    return weight

file_object = open("/u/cs246/data/adult/a7a.test")
lines = file_object.readlines()
data_list = [word.strip() for word in lines]


X1= np.array(new_data_list_combine)
Y1= np.array(labels)
weight = perceptron_train(X,Y,1)
w=weight.tolist()
w =[w[-2]]+ w[0:-2]+ [w[-1]]
def perceptron_test(X,Y,w):
    weight=np.zeros(len(X[0]))  
    learning_rate=1
    epoches = 1

    for t in range(epoches):
        err_count=0
    
        for i ,p in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                err_count += 1  
        
        accurate_rate = 1-err_count/len(X)       
        print('Test Accuracy :',accurate_rate)       

    return accurate_rate


def main():
    if '--iterations' in sys.argv:
        ep = int(sys.argv[sys.argv.index('--iterations')+1])
        perceptron_test(X1,Y1,weight)
        print('Feature weights (bias last): '," ".join(str(x) for x in w))
if __name__ == "__main__":
    main()
        


