# B.Wierzchowski
# python .\program2

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
import rich.traceback
from rich.progress import track
import os
import imageio
import scipy as sp
from scipy import integrate
from scipy.linalg import lu_factor, lu_solve, solve
import matplotlib

matplotlib.use("Agg")
rich.traceback.install()
console = Console()

J=200
delta_t = 0.002
k0 = 10
delta_k = 1
delta_x = 30/(J-1)

i = 1j 
x_tics = np.linspace(-5,25, J)


M = np.zeros((J,J),dtype=complex)
V0 = 1.1
sigma = 0.5

V_J = V0 * np.exp(-(x_tics - 10)**2/sigma**2)

for k in range(V_J.shape[0]):
    if V_J[k] < 0.01:
        V_J[k] = 0

a_j = -i*delta_t/(2*delta_x**2)
b_j = 1 + i*delta_t/2 * (2/delta_x**2 + V_J)
c_j = a_j

M[0][0] = b_j[0]
M[0][1] = c_j

M[J-1][J-2] = a_j
M[J-1][J-1] = b_j[J-1]

for k in range(1,J-1):
    M[k][k] = b_j[k]
    M[k][k-1] = a_j
    M[k][k+1] = c_j


#y_ticks = np.zeros(J,dtype=complex)

y_ticks = delta_k**0.5/np.pi**0.25 * np.exp(-x_tics**2 * delta_k**2/2) * np.exp(i*k0*x_tics)

for k in range(y_ticks.shape[0]):
    if np.abs(y_ticks[k]) < 0.01:
        y_ticks[k] = 0
        
y_ticks_abs = np.abs(y_ticks)

area0 = integrate.simpson(y_ticks_abs, x_tics)

plt.plot(x_tics, y_ticks_abs)
plt.plot(x_tics, V_J)
plt.ylim((0,1.5))
plt.xlabel('x [a.u.]')
plt.ylabel(r"$|\psi|$ [a.u.]")
plt.savefig(f'data/{0}.png')
plt.close('all')

filenames = []
filenames.append(f'{0}.png')

for k in track(range(100)):
    # plot the line chart
    
    y_ticks = solve(M, y_ticks)

    for k in range(y_ticks.shape[0]):
        if np.abs(y_ticks[k]) < 0.01:
            y_ticks[k] = 0
    
    y_ticks_abs = np.abs(y_ticks)
    # area = integrate.simpson(y_ticks_abs, x_tics)
    
    # y_ticks = area0/area * y_ticks
    
    # y_ticks_abs = np.abs(y_ticks)
    
    plt.plot(x_tics, y_ticks_abs)
    plt.plot(x_tics, V_J)
    plt.ylim((0,1.5))
    plt.xlabel('x [a.u.]')
    plt.ylabel(r"$|\psi|$ [a.u.]")
    
    # create file name and append it to a list
    filename = f'{k+1}.png'
    filenames.append(filename)
    
    # save frame
    plt.savefig(f'data/{filename}')
    plt.close('all')
# build gif
console.print('Przygotowywanie pliku .gif', style="bold blue")
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(f'data/{filename}')
        writer.append_data(image)
        
# Remove files
# for filename in set(filenames):
#     os.remove(f'data/{filename}')
