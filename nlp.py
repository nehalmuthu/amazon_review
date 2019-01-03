#correct

import pandas as pd

import numpy as np
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
import collections,re
#import naiveBayesClassifier as nb
s=pd.read_csv("train.ft.txt",sep="\n",nrows=1000)
x=s.values

y=np.zeros((1000,2))
for i in range(1000):
    c=' '.join(x[i])
    if 'label__1' in c:
        y[i]=[1,0]
        x[i]=c.lstrip("__label__1")
    elif 'label__2' in c:
        y[i]=[0,1]
        x[i]=c.lstrip("__label__2")

#
##
stop_words=set(stopwords.words("english"))
#punt=string.punctuation
bag=[]
word1=[]
for i in range(1000):
    c=' '.join(x[i])
    c=word_tokenize(c.lower())
    for w in c:
        if w not in stop_words:
            word1.append(w)


words=' '.join(word1)
words = word_tokenize(words)
words=[word.lower() for word in words if word.isalpha()] 



bag=nltk.FreqDist(words)
a=bag.most_common(10000)


b=[]
for i in range(len(a)):
    b.append(a[i][0])
    

i=0
xans=np.zeros((1000,len(b)))

#for w in b:
for i in range(1000):
    temp=[]
    c=' '.join(x[i])
    c=word_tokenize(c.lower())
    for w1 in c:
        if w1 not in stop_words:
            if w1.isalpha():
                temp.append(w1)
    j=0
    for w in b:
        if w in temp:
            xans[i][j]=1
        j=j+1
                        

    




#def fun1(k):
#    return np.tanh(k)
#def sigma1(k):
#    return np.round(np.tanh(k))

#def diffsigma1(k):
#    return 1-np.square(np.tanh(k))
def sigma2(k):
    return np.round(1/(1+np.exp(-k)))
def fun(k):
    return 1/(1+np.exp(-k))
def diffsigma2(k):
    return np.multiply(fun(k),1-fun(k))

nlayer=2
m=7998
n=[7998,3,2]

w=list()
for i in range(nlayer):
    w1=np.random.rand(n[i+1],n[i])*0.01
    w1=np.asmatrix(w1)
    w.append(w1)

b=list()
for i in range(nlayer):
    b1=np.random.rand(1,n[i+1])*0.01
    b1=np.asmatrix(b1)
    b.append(b1)

for i in range(1000):
    a=list()
    a.append(xans)
    z=list()
    dz=list()
    da=list()
    dw=list()
    db=list()
    for l in range(nlayer):
        z1=np.dot(a[l],w[l].transpose())+b[l]
        z.append(z1)
        if l!=nlayer-1:
            a1=fun(z1)
        else: 
            a1=sigma2(z1)
        a.append(a1)
        
    for l in range(nlayer):
        if l==0:
            dz1=a[nlayer]-y
        else:
            dz1=np.multiply(da[l-1],diffsigma2(z[nlayer-l-1]))
        dz.append(dz1)
        dw1=np.dot(dz[l].transpose(),a[nlayer-l-1])/m
        dw.append(dw1)
        
        db1=np.sum(dz[l],axis=0)/m
        db.append(db1)
        da1=np.dot(dz[l],w[nlayer-l-1])
        da.append(da1)
        
    for l in range(nlayer):
        w[l]=w[l]-0.8*dw[nlayer-l-1]
        b[l]=b[l]-0.8*db[nlayer-l-1]
            
    




 
a=list()
a.append(xans)
z=list()
dz=list()
da=list()
dw=list()
db=list()
          
for l in range(nlayer):
        z1=np.dot(a[l],w[l].transpose())+b[l]
        z.append(z1)
        if l!=nlayer-1:
            a1=fun(z1)
        else: 
            a1=sigma2(z1)
        a.append(a1)
        






#// fixed layers

#m=7998
#n0=m
#n1=3
#n2=2
#
#def roun(a,t):
#    return a>=t
#
#def fun1(k):
#    return np.tanh(k)
#def sigma1(k):
#    return np.round(np.tanh(k))
#
#def diffsigma1(k):
#    return 1-np.square(np.tanh(k))
#def sigma2(k):
#    return roun(1/(1+np.exp(-k)),0.5)
#def fun(k):
#    return 1/(1+np.exp(-k))
#def diffsigma2(k):
#    return np.multiply(fun(k),1-fun(k))
#
#w1=np.random.rand(n1,n0)*0.01
#w1=np.asmatrix(w1)
#b1=np.random.rand(1,n1)*0.01
#
#b1=np.asmatrix(b1)
#
#w2=np.random.rand(n2,n1)*0.01
#w2=np.asmatrix(w2)
#b2=np.random.rand(1,n2)*0.01
#b2=np.asmatrix(b2)
#
##for j in range(10):
#    
#for i in range(1000):
#    c1=b1.copy()
#    z1=np.dot(xans,w1.transpose())+c1
#    a1=fun(z1)  
#    c2=b2.copy()
#    z2=np.dot(a1,w2.transpose())+c2
#    a2=sigma2(z2)
#    dz2=a2-y
#    dw2=np.dot(dz2.transpose(),a1)/m
#    db2=np.sum(dz2,axis=0)/m
#    dz1=np.multiply(np.dot(dz2,w2),diffsigma2(z1))
#    dw1=np.dot(dz1.transpose(),xans)/m
#    db1=np.sum(dz1,axis=0)/m
#    w1=w1-0.8*dw1
#    b1=b1-0.8*db1
#    w2=w2-0.8*dw2
#    b2=b2-0.8*db2
#
#
#
#
#
#t1=np.dot(xans,w1.transpose())+b1
#t2=fun(t1)  
#t3=np.dot(t2,w2.transpose())+b2
#t4=sigma2(t3)
#

                

