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

def awgn(s, SNRdB, L=1):
    """
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's'
    to generate a resulting signal vector 'r' of specified SNR in dB. 
    It also returns the noise vector 'n' that is added to the signal 's' and the 
    power spectral density N0 of noise added
    
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L=1
    
    Returns:
        r : received signal vector (r=s+n)
        
        https://github.com/hrtlacek/SNR/blob/main/SNR.ipynb
    """
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1: #if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: #multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) #if s is a matrix [MxN]
    N0=P/gamma #Find the noise spectral density
    if not isinstance(s, complex): #check if input is real/complex object type
        n = np.sqrt(N0/2)*np.random.standard_normal(s.shape) #computed noise
    else:
        n = np.sqrt(N0/2)*np.random.standard_normal(s.shape)+1j*np.random.standard_normal(s.shape)
    r = s + n #received signal
    
    return r


#%% Data of system
def get_dataset(system, noise=1, SNR=40):
    
    if system == 'tractor':
        # Tractor dynamics
        # System variables
        lf = 1.95 #m
        lr = 1 #m
        m = 9500 #kg
        I = 18500 #kg-m**2
        kf = 1200 #N/Deg
        kr = 2500 #N/Deg
        v = 2 #m/s
        Ts = 0.1 #s
    
        # Experiment variables
        samples = 100
        signal_strength = 1
        SNR = 40 #dB
    
        # Dynamics and input matrix of a SIMO system
        A = np.array([[-2*(kf + kr)/(m*v), (-2*(kf*lf-kr*lr)-m*v**2)/(m*v**2)], 
                      [-2*(kf*lf-kr*lr)/I, -2*(kr*lf**2+kr*lr**2)/(I*v)]])
        B = np.array([[(2*kf)/(m*v)],
                      [(2*kf*lf)/I]])
        x0 = np.array([[0],
                       [0]])
    
        DataX = x0
        DataU = 3*np.sin(np.linspace(0,Ts*samples,samples))
    
        for i in range(samples):
            u =  DataU[i]
            DataXk1 = A @ DataX[:,i].reshape(2,1) + (B @ [u]).reshape(2,1)
            DataX = np.concatenate((DataX, DataXk1),axis=1)
            
        if noise:
            dataset = {
                'X' :  DataX,
                'Y' :  np.array([DataU]),
                'X_noise' :  awgn(DataX,SNR),
                'Y_noise' :  awgn(np.array([DataU]),SNR),
                }
        else:
            dataset = {
            'X' :  DataX,
            'Y' :  np.array([DataU]),
            }
        
    elif system == 'random':
        dataset = create_system(25,10)
        
    elif system == 'buildsys':
        data_path = 'C:/Users/Jens/Documents/GitHub/datadrivenBMS/Dynamic_Mode_Decompostion/assets/data/data.pkl'
        with open(data_path, 'rb') as file:
            dataset = pickle.load(file)
   
    return dataset
   
#%%
dataset = get_dataset('tractor', noise=1, SNR=40)
print(f'Applying DMDc on systme {system}')
dmdc = DMDc(svd_rank=-1)
dmdc.fit(dataset['X'], dataset['Y'])

#%%
plt.figure(figsize=(16,6))

plt.subplot(121)
plt.title('Original system')
plt.pcolor(dataset['X'].real)
plt.colorbar()
plt.xlable('Time')
plt.ylabel('Feature number')

plt.subplot(122)
plt.title('Reconstructed system')
plt.pcolor(dmdc.reconstructed_data().real)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Feature number')

plt.show()

#%% Difference

plt.figure(figsize=(16,6))

plt.title('Error original and reconstructed system')
plt.pcolor(dmdc.reconstructed_data().real - dataset['X'].real)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Feature number')

#%%
new_u = np.exp(dataset['Y'])

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

dmdc.plot_eigs()

#%% Plot real state measurements
dataset = get_dataset('tractor', noise =0)
x = np.linspace(0,100, dataset['Y'].shape[1])
plt.scatter(x,dataset['X'][0,:-1],label='state 1')
plt.plot(x,dataset['X'][0,:-1],':',label='state 1')
plt.scatter(x,dataset['X'][1,1:], label='state 2')
plt.plot(x,dataset['X'][1,1:], ':',label='state 2')
plt.legend()
plt.grid(True)
# plt.plot(x,dataset['Y'].flatten())
plt.show()


#%%
list_SNR = range(10,60,5)
SNR = 80
epochs = 5
plt.figure(figsize=(8,6))
count = 0

list_eig = []

dataset = get_dataset('tractor', 1, SNR)

dmdc_noise = DMDc(svd_rank=-1)
dmdc_noise.fit(dataset['X_noise'], dataset['Y_noise'])
list_eig.append(dmdc_noise.eigs)

for i in range(len(list_eig)):
    plt.scatter(list_eig[i].real, list_eig[i].imag, label=SNR)

dataset = get_dataset('tractor', noise=0)
dmdc_real = DMDc(svd_rank=-1)
dmdc_real.fit(dataset['X'], dataset['Y'])
plt.scatter(dmdc_real.eigs.real, dmdc_real.eigs.imag, label='real')

theta = np.linspace(0, 2*np.pi, 100)
r = np.sqrt(1.0)
x1 = r*np.cos(theta)
x2 = r*np.sin(theta)

plt.plot(x1, x2)

plt.xlim([-1,1])
plt.ylim([-1,1])
plt.legend()
plt.grid(True, linestyle='--')

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