# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 22:30:07 2022

@author: Jens
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import pickle

from PyDMD.pydmd.dmdc import DMDc
from datadrivenBMS.Dynamic_Mode_Decompostion import sqldatabaseinteract
data_path = 'C:/Users/Jens/Documents/GitHub/datadrivenBMS/Dynamic_Mode_Decompostion/assets/data/data.pkl'

def create_sys(n, m):
    A = scipy.linalg.helmert(n, True)
    B = np.random.rand(n, n)-.5
    x0 = np.array([0.25]*n)
    u = np.random.rand(n, m-1)-.5
    snapshots = [x0]
    for i in range(m-1):
        snapshots.append(A.dot(snapshots[i])+B.dot(u[:, i]))
    snapshots = np.array(snapshots).T
    return {'snapshots': snapshots, 'u': u, 'B': B, 'A': A}

with open(data_path, 'rb') as file:
    dataset = pickle.load(file)
    
# s = np.concatenate((dataset['X'], dataset['Y']), axis=0)

dmdc = DMDc(svd_rank=-1)
dmdc.fit(dataset['X'], dataset['Y'][:,1:])

#%%
plt.figure(figsize=(16,6))

plt.subplot(121)
plt.title('Original system')
plt.pcolor(dataset['X'].real)
plt.colorbar()

plt.subplot(122)
plt.title('Reconstructed system')
plt.pcolor(dmdc.reconstructed_data().real)
plt.colorbar()

plt.show()

#%%
new_u = np.exp(dataset['Y'][:,1:])

plt.figure(figsize=(8,6))
plt.title('Reconstruct system output with scaled input')
plt.pcolor(dmdc.reconstructed_data(new_u).real)
plt.xlabel('Time')
plt.ylabel('State feature [-]')
plt.colorbar()
plt.show()

#%%
dmdc.dmd_time['dt'] = .5
new_u = np.random.rand(dataset['Y'][:,1:].shape[0], dmdc.dynamics.shape[1]-1)

plt.figure(figsize=(8,6))
plt.title('Reconstruct system output: random input, custom timescale')
plt.pcolor(dmdc.reconstructed_data(new_u).real)
plt.xlabel('Time')
plt.ylabel('State feature [-]')
plt.colorbar()
plt.show()

#%%
for eig in dmdc.eigs:
    print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))


#%%
x = np.linspace(0,dataset['X'].shape[0],dataset['X'].shape[0])
t = np.linspace(0,dataset['X'].shape[1],dataset['X'].shape[1])

for mode in dmdc.modes.T:
    plt.plot(x, mode.real)
    plt.title('Modes')
plt.show()

for dynamic in dmdc.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamics')
plt.show()