#B.Wierzchowski
#python .\program3.py

from json import load
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
import rich.traceback
from rich.progress import track
from numba import jit
import scipy as sp
import pandas as pd

@jit
def przelicz_hamiltionian(H_r, H_i, sigma_r, sigma_i, delta_x,kappa,x_ticks,omega,tau,N):
    
    for i in range(1,N):
        H_r[i] = - 0.5 * (sigma_r[i+1]+sigma_r[i-1] - 2 * sigma_r[i])/delta_x**2 + kappa * (x_ticks[i]-0.5)*sigma_r[i]*np.sin(omega*tau)
        H_i[i] = - 0.5 * (sigma_i[i+1]+sigma_i[i-1] - 2 * sigma_i[i])/delta_x**2 + kappa * (x_ticks[i]-0.5)*sigma_i[i]*np.sin(omega*tau)
        
    return H_r, H_i

@jit
def przelicz_parametry(delta_x, sigma_r, sigma_i, x_ticks, H_r, H_i):
    
    norma = delta_x*np.sum(np.square(sigma_r)+np.square(sigma_i))
    ex = delta_x*np.sum(x_ticks*(np.square(sigma_r)+np.square(sigma_i)))
    energia = delta_x*np.sum(sigma_r*H_r+sigma_i*H_i)
    
    return norma, ex, energia

rich.traceback.install()
console = Console()

with open("parameters.json") as f:
    parameters = load(f)

N = parameters['N']
delta_tau = parameters['delta_tau']
delta_x = 1/N
kappa = parameters['kappa']

tau = 0
omega = 3 * (np.pi)**2/2

x_ticks = np.linspace(0,1, N+1)

n = 1

energy_max_list = []

om_ = np.linspace(0.8,1.2, num = 15, endpoint=True)
om_ = np.append(om_, 1)
om_ = np.sort(om_)

step = 1

for l in om_:
    
    console.print(f'{step}/{om_.shape[0]}')
    
    energy_list = []
    time_list = []
    tau = 0
    
    sigma_r = np.sqrt(2) * np.sin(n * np.pi * x_ticks)
    sigma_i = np.zeros(x_ticks.shape)
    
    H_r = np.zeros(x_ticks.shape)
    H_i = np.zeros(x_ticks.shape)
    
    H_r, H_i = przelicz_hamiltionian(H_r, H_i, sigma_r, sigma_i, delta_x,kappa,x_ticks,omega,tau,N)
    norma, ex, energia = przelicz_parametry(delta_x, sigma_r, sigma_i, x_ticks, H_r, H_i)
    
    energy_list.append(energia)
    time_list.append(tau)
    
    for k in track(range(1, 500000)):
        
        sigma_r = sigma_r + H_i * delta_tau/2
        H_r, H_i = przelicz_hamiltionian(H_r, H_i, sigma_r, sigma_i, delta_x,kappa,x_ticks,l*omega,tau + delta_tau/2,N)
        sigma_i = sigma_i - H_r * delta_tau
        H_r, H_i = przelicz_hamiltionian(H_r, H_i, sigma_r, sigma_i, delta_x,kappa,x_ticks,l*omega,tau + delta_tau,N)
        sigma_r = sigma_r + H_i * delta_tau/2
        
        tau += delta_tau
        
        if k % 100 == 0:
            
            norma, ex, energia = przelicz_parametry(delta_x, sigma_r, sigma_i, x_ticks, H_r, H_i)
            
            # print(f'N = {norma}')
            # print(f'x = {ex}')
            # print(f'E = {energia}')
            # print('\n')
            
            energy_list.append(energia)
            time_list.append(tau)
    
    energy_max_list.append(np.max(energy_list))
    step += 1

df = pd.DataFrame(data=zip(om_, energy_max_list), columns=['om/om0', 'Max_energy'])
#print(df)
df.to_csv('data/dane.csv')