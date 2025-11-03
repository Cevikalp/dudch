#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:17:07 2021

@author: hasan
"""
import numpy as np
from matplotlib import pyplot as plt

def FindCenters(k, E=1):
    """
    Calculates "k+1" equidistant points in R^{k}.
    Args:
        k (int) dimension of the space
        E (float) expand factor 
    Returns: 
        Labels (np.array) equidistant positions in R^{k}, shape (k+1 x k)
    """
    
    Labels = np.empty((k+1, k), dtype=np.float32)
    CC = np.empty((k,k), dtype=np.float32)
    Unit_Vector = np.identity(k)
    c = -((1+np.sqrt(k+1))/np.power(k, 3/2))
    CC.fill(c)
    d = np.sqrt((k+1)/k)
    DU = d*Unit_Vector 
    Labels[0,:].fill(1/np.sqrt(k))
    Labels[1:,:] = CC + DU
    
    # Calculate and Check Distances
    # Distances = np.empty((k+1,k), dtype=np.float32)
    # for k, rows in enumerate(Labels):
    #     Distances[k,:] = np.linalg.norm(rows - np.delete(Labels, k, axis=0), axis=1)
    # # print("Distances:",Distances)    
    # assert np.allclose(np.random.choice(Distances.flatten(), size=1), Distances, rtol=1e-05, atol=1e-08, equal_nan=False), "Distances are not equal" 
    return Labels*E

# Labels = FindLabels(2)
# print("Labels:", Labels)

# fig, ax = plt.subplots()
# ax.scatter(Labels[0,0], Labels[0,1], marker='*')
# ax.scatter(Labels[1,0], Labels[1,1], marker='x')
# ax.scatter(Labels[2,0], Labels[2,1], marker='o')

# plt.show()
