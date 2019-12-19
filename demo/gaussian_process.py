#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:42:44 2019

@author: Truong Vinh Hoang
"""

import numpy as np 
from scipy.stats import norm, uniform

sigma_e = 0.1
N = 20
l = 0.5
sigma_f = 1.
f = lambda x: np.sin(x)
y_n = lambda x: f(x) + norm.rvs(size = x.size)*sigma_e

k = lambda x,y: np.exp(-(x-y)**2/2./l**2)*sigma_f

xn = uniform.rvs(size = N)*10. -5.
def observation_operator(xn):
    return y_n(xn)

yn = observation_operator(xn)

def prior_covariance (x_n,x_m):
    K = np.zeros (shape = (x_n.size, x_m.size))
    for i in range(0,x_n.size):
        for j in range (0,x_m.size):
            K[i,j] = k(x_n[i],x_m[j])
    return K

Kxnxn = prior_covariance(xn,xn) + np.eye(N)*sigma_e**2
invKxnxn = np.linalg.inv(Kxnxn)
kxx = Kxnxn[0,0]
A = np.matmul( invKxnxn, yn)
x = np.linspace(-5,5, 1000)
y_m = np.zeros(shape = x.shape)
y_sigma = np.zeros(shape = x.shape)

for i in range (0,x.size):
    Kxnx = prior_covariance(xn, np.array([x[i]])).reshape(xn.size,)
    y_m[i] = np.vdot(Kxnx, A)
    y_sigma[i] = np.sqrt(kxx - np.vdot(Kxnx,np.matmul(invKxnxn,Kxnx)) )
    
import matplotlib.pyplot as plt
plt.figure()
plt.plot(xn, yn, 'or',markersize = 10,  label = 'observation', linewidth = 2)
plt.plot(x, f(x), '-k', label = 'true', linewidth = 2)
plt.plot(x, y_m, '-b', label = 'predicted mean', linewidth = 2)
plt.plot(x, y_m-y_sigma ,'--b', label = '1 sigma', linewidth = 2)
plt.plot(x, y_m+y_sigma ,'--b', linewidth = 2)
plt.ylim([-1.5,3])
plt.legend()
plt.savefig('GP.pdf') 