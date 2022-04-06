# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 22:30:07 2022

@author: Jens

[1]H. Wang and N. Noguchi, “Real-time states estimation of a farm tractor using dynamic mode decomposition Real-time states estimation of a farm tractor using dynamic mode decomposition,” GPS Solutions, vol. 25, p. 12, Jan. 2021, doi: 10.1007/s10291-020-01051-5.

"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import pickle

from PyDMD.pydmd.dmdc import DMDc
from datadrivenBMS.Dynamic_Mode_Decompostion import sqldatabaseinteract

plt.rcParams.update({'font.size':20})
plt.rcParams['figure.figsize'] = (20,10)


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

def inputfunction(x):
    return 3*np.sin(x)

def statefunction(samples, DataU, DataX, A, B):
    for i in range(samples):
        u =  DataU[i]
        DataXk1 = A @ DataX[:,i].reshape(2,1) + (B @ [u]).reshape(2,1)
        DataX = np.concatenate((DataX, DataXk1),axis=1)
    return DataX

def get_dataset(samples, Ts, noise=1):
    
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
    signal_strength = 1

    # Dynamics and input matrix of a SIMO system
    A = np.array([[-2*(kf + kr)/(m*v), (-2*(kf*lf-kr*lr)-m*v**2)/(m*v**2)], 
                  [-2*(kf*lf-kr*lr)/I, -2*(kf*lf**2+kr*lr**2)/(I*v)]])
    B = np.array([[(2*kf)/(m*v)],
                  [(2*kf*lf)/I]])
    x0 = np.array([[0],
                   [0]])

    DataX = x0
    t = np.linspace(0,Ts*samples,samples)
    DataU = inputfunction(t)

    for i in range(samples):
        u =  DataU[i]
        DataXk1 = A @ DataX[:,i].reshape(2,1) + (B @ [u]).reshape(2,1)
        DataX = np.concatenate((DataX, DataXk1),axis=1)
        
    if noise:
        noise_X_0 = np.random.normal(0,0.01, size=DataX[0].shape)
        noise_X_1 = np.random.normal(0,0.1, size=DataX[1].shape)
        noise_Y = np.random.normal(0,0, size=DataU.shape)
        
        
        dataset = {
            'X_real' :  DataX,
            'Y_real' :  np.array([DataU]),
            'X_sens' : DataX + np.concatenate(([noise_X_0], [noise_X_1]), axis=0),
            'Y_sens' :  np.array([DataU]) + noise_Y,
            'X_noise' :  np.concatenate(([noise_X_0], [noise_X_1]), axis=0),
            'Y_noise' : noise_Y,
            }
    else:
        dataset = {
        'X' :  DataX,
        'Y' :  np.array([DataU]),
        }
   
    return dataset
   
lf = 1.95 #m
lr = 1 #m
m = 9500 #kg
I = 18500 #kg-m**2
kf = 1200 #N/Deg
kr = 2500 #N/Deg
v = 2 #m/s
Ts = 0.1 #s    
   
A = np.array([[-2*(kf + kr)/(m*v), (-2*(kf*lf-kr*lr)-m*v**2)/(m*v**2)], 
              [-2*(kf*lf-kr*lr)/I, -2*(kf*lf**2+kr*lr**2)/(I*v)]])
B = np.array([[(2*kf)/(m*v)],
              [(2*kf*lf)/I]])
x0 = np.array([[0],
               [0]])

A_eig_c = np.linalg.eig(A) # -0.39+-0.13
A_eig_d = np.exp(A_eig_c[0])**Ts #0.962+-0.012

#%%
Ts = 0.1
samples = 100
dataset = get_dataset(samples=samples, Ts=Ts, noise=1)

t = np.linspace(0,Ts*samples,samples)

plt.figure(figsize=[20,15])
plt.subplot(311)
plt.title('Original system state 1 (vehicle angle)')
plt.scatter(t,dataset['X_sens'][0,:-1])
plt.plot(t, dataset['X_real'][0,:-1], ':r', label='real')
plt.xlabel('Time [s]')
plt.ylabel('b [rad]')

plt.subplot(312)
plt.title('Original system state 2 (vehicle angular speed)')
plt.scatter(t,dataset['X_sens'][1,:-1])
plt.plot(t, dataset['X_real'][1,:-1], ':r', label='real')
plt.xlabel('Time [s]')
plt.ylabel('w [rad/s]')

plt.subplot(313)
plt.title('Original system input (vehicle steering angle)')
plt.scatter(t,dataset['Y_sens'][0])
plt.plot(t, dataset['Y_real'][0], ':r', label='real')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid(True)

#%%
print(f'Applying DMDc on tractor system')
dmdc = DMDc(svd_rank=2)
dmdc.fit(dataset['X_sens'], dataset['Y_sens'])

plt.figure(figsize=(16,6))
plt.subplot(121)
plt.title('Original system')
plt.pcolor(dataset['X_sens'].real)
plt.colorbar()
plt.xlabel('Time')
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
mean_rec_0 = np.mean(dmdc.reconstructed_data()[0].real)
var_rec_0 = np.std(dmdc.reconstructed_data()[0].real)
mean_ds_0 = np.mean(dataset['X_sens'][0].real)
var_ds_0 = np.std(dataset['X_sens'][0].real)

mean_rec_1 = np.mean(dmdc.reconstructed_data()[1].real)
var_rec_1 = np.std(dmdc.reconstructed_data()[1].real)
mean_ds_1 = np.mean(dataset['X_sens'][1].real)
var_ds_1 = np.std(dataset['X_sens'][1].real)

err_0 = (dmdc.reconstructed_data()[0].real-mean_rec_0)/var_rec_0 - (dataset['X_sens'][0].real-mean_ds_0)/var_ds_0
err_1 = (dmdc.reconstructed_data()[1].real-mean_rec_1)/var_rec_1 - (dataset['X_sens'][1].real-mean_ds_1)/var_ds_1

err_mat = np.concatenate(([err_0],[err_1]))
plt.pcolor(err_mat)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Feature number')

#%% 
# Compare the fitted DMDc with the original dataset

# new_u = np.exp(dataset['Y'])
new_u = dataset['Y_real']

plt.figure(figsize=(20,10))
plt.subplot(131)
plt.title('Original system measured state')
plt.pcolor(dataset['X_real'])
plt.xlabel('Time')
plt.ylabel('State feature [-]')
plt.colorbar()

plt.subplot(132)
plt.title('Reconstruct system measured state with original input')
plt.pcolor(dmdc.reconstructed_data(new_u).real)
plt.xlabel('Time')
plt.ylabel('State feature [-]')
plt.colorbar()

plt.subplot(133)
plt.title('Reconstruction error')
plt.pcolor(dmdc.reconstructed_data(new_u).real - dataset['X_real'])
plt.xlabel('Time')
plt.ylabel('State feature [-]')
plt.colorbar()
plt.show()

#%%
# Different time frequency

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
# Eigenvalues for different noise levels

Ts = 0.1
samples = 100
dataset = get_dataset(samples=samples, Ts=Ts, noise=1)

print('Eigenvalues for original system')
dmdc_origin = DMDc(svd_rank=2)
dmdc_origin.fit(dataset['X_real'], dataset['Y_real'])
for eig in dmdc_origin.eigs:
    print('DMDc origin eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))

print('Eigenvalues for noisy system')
dmdc_noise = DMDc(svd_rank=2)
dmdc_noise.fit(dataset['X_sens'], dataset['Y_sens'])
for eig in dmdc_noise.eigs:
    print('DMDc noisy eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))

dmdc_origin.plot_eigs()
dmdc_noise.plot_eigs()

#%% Frobenius norm

std_list = np.arange(0,0.2,0.001)
parm_list = ['X_0', 'X_1', 'Y']
samples = 100
Ts = 0.1

parm_sol = []
for parm in parm_list:
    
    print(f'start with parameter {parm}')
    DataX = x0
    t = np.linspace(0,Ts*samples,samples)
    DataU = 3*np.sin(t)

    for i in range(samples):
        u =  DataU[i]
        DataXk1 = A @ DataX[:,i].reshape(2,1) + (B @ [u]).reshape(2,1)
        DataX = np.concatenate((DataX, DataXk1),axis=1)

    std_sol = []
    for std in std_list:
    
        print(f'start with standard diviation {std}')
        if parm == 'X_0':
            noise_X_0 = np.random.normal(0,std, size=DataX[0].shape)
            noise_X_1 = np.random.normal(0,0, size=DataX[1].shape)
            noise_Y = np.random.normal(0,0, size=DataU.shape)
        
        if parm =='X_1':
            print('Check X1')
            noise_X_0 = np.random.normal(0,0, size=DataX[0].shape)
            noise_X_1 = np.random.normal(0,std, size=DataX[1].shape)
            noise_Y = np.random.normal(0,0, size=DataU.shape)
        
        if parm =='Y':
            print('Check Y')
            noise_X_0 = np.random.normal(0,0, size=DataX[0].shape)
            noise_X_1 = np.random.normal(0,0, size=DataX[1].shape)
            noise_Y = np.random.normal(0,std, size=DataU.shape)
     
        dataset = {
            'X_real' :  DataX,
            'Y_real' :  np.array([DataU]),
            'X_sens' : DataX + np.concatenate(([noise_X_0], [noise_X_1]), axis=0),
            'Y_sens' :  np.array([DataU]) + noise_Y,
            'X_noise' :  np.concatenate(([noise_X_0], [noise_X_1]), axis=0),
            'Y_noise' : noise_Y,
            }
        
        dmdc = DMDc(svd_rank=2)
        dmdc.fit(dataset['X_sens'], dataset['Y_sens'])
        frobnorm = np.linalg.norm(A_eig_c[0][0]-dmdc.eigs[0])
        
        # eig_sol.append(dmdc.eigs[0])
        std_sol.append(frobnorm)
        
    parm_sol.append(std_sol)
     
plt.figure(figsize=[30,30])
plt.subplot(311)
plt.title('state 1  noise')
plt.plot(std_list, parm_sol[0], label='real')
plt.xlabel('noise standard deviation')
plt.ylabel('Frobenius norm')
plt.grid(True)

plt.subplot(312)
plt.title(' state 2  noise')
plt.plot(std_list, parm_sol[1], label='real')
plt.xlabel('noise standard deviation')
plt.ylabel('Frobenius norm')
plt.grid(True)

plt.subplot(313)
plt.title('input noise')
plt.plot(std_list, parm_sol[2])
plt.xlabel('noise standard deviation')
plt.ylabel('Frobenius norm')
plt.legend()
plt.grid(True)

#%%
# eig_continuous = np.linalg.eig(A)
# eig_discrete = np.exp(eig_continuous[0][0])**Ts
# print('Original eigenvalue {}: distance from unit circle {}'.format(eig_discrete, np.abs(eig_discrete.imag**2+eig_discrete.real**2 - 1)))
epochs = 100
 
plt.figure(figsize=[10,10])

for epoch in range(epochs):
    
    dataset = get_dataset(100, 0.1)
    dmdc = DMDc(svd_rank=-1)
    dmdc.fit(dataset['X_sens'], dataset['Y_sens'])

    dmdc_eig_d = np.exp(dmdc.eigs[0])**Ts

    err = np.linalg.norm(eig_continuous[0][0] - dmdc.eigs[0])
    plt.plot(dmdc_eig_d.real, dmdc_eig_d.imag, 'r*')
    
plt.plot(A_eig_d[0].real, A_eig_d[0].imag, '^', markersize=16,label='real')
circle = plt.Circle((0,0), 1, color='r', alpha=0.2)
plt.gca().add_patch(circle)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.legend()
#%%
Ts = 0.1
samples = 100
dataset = get_dataset(samples=samples, Ts=Ts, noise=1)

t = np.linspace(0,Ts*samples,samples)

dataset = get_dataset(100, 0.1)
dmdc = DMDc(svd_rank=-1)
dmdc.fit(dataset['X_sens'], dataset['Y_sens'])

dmdc_eig_d = np.exp(dmdc.eigs[0])**Ts

err = np.linalg.norm(eig_continuous[0][0] - dmdc.eigs[0])
print(f'Error between positive only continuous value: {err}')


# Plot eigenvalues 
plt.figure(figsize=[20,50])

plt.subplot(621)
plt.title('Original system state 1 (vehicle angle)')
plt.scatter(t,dataset['X_sens'][0,:-1])
plt.plot(t, dataset['X_real'][0,:-1], ':r', label='real')
plt.xlabel('Time [s]')
plt.ylabel('b [rad]')

plt.subplot(622)
plt.title('Original system state 2 (vehicle angular speed)')
plt.scatter(t,dataset['X_sens'][1,:-1])
plt.plot(t, dataset['X_real'][1,:-1], ':r', label='real')
plt.xlabel('Time [s]')
plt.ylabel('w [rad/s]')

plt.subplot(623)
plt.title('Original system input (vehicle steering angle)')
plt.scatter(t,dataset['Y_sens'][0])
plt.plot(t, dataset['Y_real'][0], ':r', label='real')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid(True)

plt.subplot(624)
plt.title('Eigenvalue')
plt.plot(A_eig_d[0].real, A_eig_d[0].imag, '^', markersize=16,label='real')
plt.plot(dmdc_eig_d.real, dmdc_eig_d.imag, '*', label='reconstructed')
circle = plt.Circle((0,0), 1, color='r', alpha=0.2)
plt.gca().add_patch(circle)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.legend()

plt.subplot(625)
plt.title('Eigenvalue')
plt.plot(A_eig_d[0].real, A_eig_d[0].imag, '^', markersize=16,label='real')
plt.plot(dmdc_eig_d.real, dmdc_eig_d.imag, '*', label='reconstructed')
plt.xlabel('Real')
plt.ylabel('Imag')
plt.legend()

#%%

Ts = 0.1
samples = 100
x0 = np.array([[0],
               [0]])

t = np.linspace(0,Ts*samples,samples+1)

dataset = get_dataset(samples=samples, Ts=Ts, noise=1)


dataset = get_dataset(samples, Ts)
dmdc = DMDc(svd_rank=2)
dmdc.fit(dataset['X_sens'], dataset['Y_sens'])

dmdc.original_time['dt'] = dmdc.dmd_time['dt'] = t[1] - t[0]
dmdc.original_time['t0'] = dmdc.dmd_time['t0'] = t[0]
dmdc.original_time['tend'] = dmdc.dmd_time['tend'] = t[-1]

plt.figure(figsize=[30,20])
plt.subplot(2,1,1)
plt.title('DMDc reconstruction of state 1')
plt.plot(dmdc.original_timesteps, dataset['X_sens'][0,:], '.', label='measurements')
plt.plot(dmdc.original_timesteps, 
         statefunction(100, inputfunction(t), x0, A, B)[0,:], 
         '-', 
         label='original function')
plt.plot(dmdc.dmd_timesteps, 
          dmdc.reconstructed_data().real[0,:], 
          '--', 
          label='DMDc output')
plt.xlabel('Time [s]')
plt.ylabel('b [rad]')
plt.legend()

plt.subplot(2,1,2)
plt.title('DMDc reconstruction of state 2')
plt.plot(dmdc.original_timesteps, dataset['X_sens'][1,:], '.', label='measurements')
plt.plot(dmdc.original_timesteps, 
         statefunction(100, inputfunction(t), x0, A, B)[1,:], 
         '-', 
         label='original function')
plt.plot(dmdc.dmd_timesteps, 
          dmdc.reconstructed_data().real[1,:], 
          '--', 
          label='DMDc output')
plt.xlabel('Time [s]')
plt.ylabel('w [rad/s]')
plt.legend()
plt.show()

#%%
samples = 50
t = np.linspace(0,Ts*samples,samples+1)
dmdc.dmd_time['tend'] = t[-1]

fig = plt.figure()
plt.plot(dmdc.original_timesteps, 
         dataset['X_sens'][0,:], 
         '.', 
         label='measurements')
plt.plot(t, 
         statefunction(samples, inputfunction(t), x0, A, B)[0,:],
         '-', 
         label='original function')
plt.plot(dmdc.dmd_timesteps, 
         dmdc.reconstructed_data().real[0,:],
         '--',
         label='DMDc output')
plt.legend()
plt.show()



#%%

x = np.linspace(0,dataset['X_real'].shape[0],dataset['X_real'].shape[0])
t = np.linspace(0,dataset['X_real'].shape[1],dataset['X_real'].shape[1])

for mode in dmdc_origin.modes.T:
    plt.plot(x, mode.real)
    plt.title('Modes')
plt.show()

for dynamic in dmdc_origin.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamics')
plt.show()