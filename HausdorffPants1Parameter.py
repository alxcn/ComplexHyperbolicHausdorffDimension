# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:03:16 2017

@author: @alxcn
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
# The following algorithm provides the Hausdorff dimension of the limit set for a 1-parameter family of Theta-Schottky groups on PU(2,1)
# INPUT
# n = Number of desired groups
# k = Maximal depth on the Word-tree
# OUTPUT
# [t,d] = array of angles t and Hausdorff dimension of the limit set of G_t

def PerronFrobenius(p,A,y):
    import numpy as np
    cnt=0
#    y=np.ones([p],dtype=float)
#    y=y*(1.00/float(p))
    M=np.linalg.norm(y)
    y=y*(1.00/M)
    ynew=np.zeros([p],dtype=float)
    while (cnt<1000):
        ynew=np.dot(A,y)
        L=np.linalg.norm(ynew)
        ynew=ynew*(L**(-1.00))
        if np.allclose(y,ynew,1e-15,1e-15):
            break
        else:
            y=ynew
            cnt=cnt+1
    alpha=np.dot(np.dot(ynew,A),ynew)/np.dot(ynew,ynew)
    return alpha, y

def Hausdorff(levmax,t,d):
    import numpy as np
    import math as m
    N=3*(2**(levmax-1))
    N1=3*(2**(levmax))
    tagpoints=np.zeros([3000000,2], dtype=complex)
    w=np.zeros([N1,2],dtype=complex)
    words=np.zeros([3000000],dtype=object)
    wordsN=np.zeros([N],dtype=object)
    e1=(1.00+m.sqrt(3.00)*1.00j)/2.00
    e2=(1.00-m.sqrt(3.00)*1.00j)/2.00
    centros=[[-1.00/m.cos(t*m.pi),0.00],[1.00/m.cos(t*m.pi)*e1,0.00],[(1.00/m.cos(t*m.pi))*e2,0.00]]
    radios=np.zeros([3],dtype=float)
    radios=[m.tan(t*m.pi), m.tan(t*m.pi),m.tan(t*m.pi)]
    tagpoints[0:3]=[[-1.00,0.00],[e1,0.00],[e2,0.00]]
    #tagpoints[0:3]=[[-1.00/m.cos(t*m.pi),0.00],[1.00/m.cos(t*m.pi)*e1,0.00],[(1.00/m.cos(t*m.pi))*e2,0.00]]
    inv=np.array([1,2,3])
    tag=np.zeros([3000000],dtype=int)
    num=np.zeros([levmax+2],dtype=int)
    for i in range(0,3):
        tag[i]=i+1
    words[0]='1'
    words[1]='2'
    words[2]='3'
    num[0]=1
    num[1]=4
    for lev in range(2,levmax+2):
        inew=num[lev-1]
        for j in range(1,4):
            for iold in range(num[lev-2],num[lev-1]):
                if j==inv[tag[iold-1]-1]:
                    continue
                z=float(abs(tagpoints[iold-1][0]-centros[j-1][0])**4.00+(tagpoints[iold-1][1]-centros[j-1][1]-2*(np.conjugate(tagpoints[iold-1][0])*centros[j-1][0]).imag)**2)
                u=complex(abs(tagpoints[iold-1][0]-centros[j-1][0])**2+1j*(tagpoints[iold-1][1]-centros[j-1][1]-2*(np.conjugate(tagpoints[iold-1][0])*centros[j-1][0]).imag))
                tagpoints[inew-1][0]=centros[j-1][0]+(-1.00*radios[j-1]**2)*(tagpoints[iold-1][0]-centros[j-1][0])*u*(z**(-1.00))
                tagpoints[inew-1][1]=centros[j-1][1]+(z**(-1.00))*(-1.00*(radios[j-1]**4)*(tagpoints[iold-1][1]-centros[j-1][1]-2*(np.conjugate(tagpoints[iold-1][0])*centros[j-1][0]).imag)+(radios[j-1]**2)*2*((np.conjugate((tagpoints[iold-1][0]-centros[j-1][0])*u)*centros[j-1][0]).imag))
                words[inew-1]=words[iold-1]
                words[inew-1]=''.join([words[j-1],words[inew-1]])
                tag[inew-1]=j
                inew+=1
        num[lev]=inew
    for i in range(0,num[levmax+1]-num[levmax]):
        w[i]=tagpoints[num[levmax]+i-1]
    for i in range(0,num[levmax]-num[levmax-1]):
        wordsN[i]=words[num[levmax-1]+i-1]
    T=np.zeros([N,N],dtype=float)
    if levmax==1:
        cont=0
        for i in range(0,N):
            for j in range(0,N):
                if i==j:
                    continue
                else:
                    z=(abs(radios[i])**4)/(abs(w[cont][0]-centros[i][0])**4+(w[cont][1]-centros[i][1]-2*(np.conjugate(w[cont][0])*centros[i][0]).imag)**2)
                    T[i][j]=m.sqrt(abs(z))**-1.00
                    cont=cont+1
    else:
        cont=0
        for i in range(0,N):
            for j in range(0,N):
                if wordsN[i][1:levmax]==wordsN[j][0:levmax-1]:
                    for k in range(0,3):
                        if (wordsN[i][0]==words[k]):
                            z=(abs(radios[k])**4)/(abs(w[cont][0]-centros[k][0])**4+(w[cont][1]-centros[k][1]-2*(np.conjugate(w[cont][0])*centros[k][0]).imag)**2)
                            T[i][j]=abs(z)**-1.00
                            cont=cont+1
    Td=np.zeros([N,N],dtype=float)
    x=np.ones([N],dtype=float)
    x=x*(1.00/N)
    x0=np.ones([N],dtype=float)
    x0=x0*(1.00/N)
    aNew=d
    while cont<350:
        Td=np.power(T,aNew)
        a,x=PerronFrobenius(N,Td,x)
        if abs(a-1.00)<1e-15:
            break
        else:
            d0=d + 0.1
            TdEpsilon=np.power(T,d0)
            aEpsilon,yEpsilon=PerronFrobenius(N,TdEpsilon,x0)
            Der=(aEpsilon-a)/(0.01)
            aNew=d+(1.00-a)/(Der)
            d=aNew
            if abs(d-1.00)<1e-15:
                break
            else:
                cont=cont+1
    return(d)
    
n=input('Provide the Number maximal of groups ')
k=input('Maximal depth on the Word-tree')
angulos=np.zeros([n+1],dtype=float)
dimension=np.zeros([n+1],dtype=float)
for i in range(1,n+1):
    t=0.00+(float(i)/(3.00*float(n)))
    angulos[i]=t
    if float(i)/float(n)<0.5:
        d=Hausdorff(k,t,0.5)
    else:
        d=Hausdorff(k,t,1.00)
    dimension[i]=d

plt.plot(angulos,dimension, linewidth=2.0)
plt.ylabel('Hausdorff dimension')
plt.xlabel('Angle < Pi/3')
plt.show()
