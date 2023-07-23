# B.Wierzchowski
# python .\program1_final.py -ifile input.txt -ofile output.txt -oxyzfile output.xyz

import numpy as np
from numba import jit, njit
import sys
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from rich.console import Console
import rich.traceback
from rich.progress import track

@jit()
def przelicz_hamiltonian(N, M, V, p_i):

    H = V

    for i in range(N):
        H += (np.linalg.norm(p_i[i]) ** 2)/(2 * M)
    
    return H

@jit()
def przelicz_temperatury(N, M, p_i):

    E_kin_i = np.zeros(N)
    for i in range(N):
        E_kin_i[i] = (np.linalg.norm(p_i[i]) ** 2) / (2*M)

    T = 2 * np.sum(E_kin_i) / (3 * N * 8.31e-3)

    return T

@jit()
def przelicz_ped(tau, p_i, F_i):

    return (p_i + 0.5 * F_i * tau)

@jit()
def przelicz_polorzenie(M, tau, r_i, p_i):

    return (r_i + (1/M) * p_i * tau)

@jit()
def przelicz_sily(N, e, R, L, f, r_i, F_i, F_S_i, F_P_i_j):

        Vs_ri = np.zeros(N)
        Vp_rij = np.zeros((N,N))
        #F_S_i = np.array([np.zeros(3) for i in range(N)])
        #F_P_i_j = np.array([[np.zeros(3) for x in range(N)] for y in range(N)])

        for i in range(0,N):
            norm_ri = np.linalg.norm(r_i[i])
            if norm_ri < L:
                Vs_ri[i] = 0
            else:
                Vs_ri[i] = 0.5 * f * (norm_ri - L)**2

        for i in range(0,N):
            for j in range(0,i):
                norm_r_ij = np.linalg.norm((r_i[i] - r_i[j]))
                Vp_rij[i][j] = e * ((R/norm_r_ij)**12 - 2*(R/norm_r_ij)**6)
                Vp_rij[j][i] = Vp_rij[i][j]
        
        V = 0
        for i in range(1, N):
            for j in range(0, i):
                V += Vp_rij[i][j]

        V += np.sum(Vs_ri)

        for i in range(0,N):
            norm_ri = np.linalg.norm(r_i[i])
            if norm_ri < L:
                F_S_i[i] = np.zeros(3)
            else:
                F_S_i[i] = f * (L - norm_ri) * r_i[i]/norm_ri

        for i in range(0,N):
            for j in range(0,i):
                norm_r_ij = np.linalg.norm((r_i[i] - r_i[j]))
                F_P_i_j[i][j] = 12 * e * ((R/norm_r_ij)**12 - (R/norm_r_ij)**6) * (r_i[i] - r_i[j])/norm_r_ij**2
                F_P_i_j[j][i] = - F_P_i_j[i][j]

        for i in range(0,N):
            F_i[i] = F_S_i[i] + np.sum(F_P_i_j,axis=1)[i]

        sum_F_S_i = 0

        for i in range(0,N):
            norm_F_S_i = np.linalg.norm(F_S_i[i])
            sum_F_S_i += norm_F_S_i

        P = 1/(4 * np.pi * L**2) * sum_F_S_i

        return(F_i, V, P)

random.seed(10)

rich.traceback.install()

parser = argparse.ArgumentParser(description='Program 1')
parser.add_argument('-ifile', help='name of the input file', default='input.txt')
parser.add_argument('-ofile', help='name of the output file', default='output.txt')
parser.add_argument('-oxyzfile', help='name of the output xyz file', default='output.xyz')

args = parser.parse_args()

file_parameters = args.ifile
file_output = args.ofile
file_xyz = args.oxyzfile

parameters = {}

with open(f'data/{file_parameters}', 'r') as file:
    for line in file:
        value, name = line.strip().split('#')
        parameters[str(name).strip()] = float(value.strip())

n = int(parameters['n'])
M = parameters['m']
e = parameters['e']
R = parameters['R']
f = parameters['f']
L = parameters['L']
a = parameters['a']
T_0 = parameters['T_0']
tau = parameters['tau']
S_0 = int(parameters['S_o'])
S_d = int(parameters['S_d'])
S_out = parameters['S_out']
S_xyz = parameters['S_xyz']

N = n**3

console = Console()
console.clear()
console.rule(f'Symulacja kryształu Argonu dla n = {n} oraz T = {T_0} K.')
console.print()

L_min = 1.22 * a * (n - 1)

if L > L_min:
    console.print("Naczynie odpowiedniej wielkości.", style="bold green")
else:
    console.print("Naczynie jest za małe!!!", style="bold red")
    sys.exit()

console.print("Obliczanie warunków początkowych.", style="bold yellow")

b0 = np.array([a, 0, 0])
b1 = np.array([a/2, a*math.sqrt(3)/2, 0])
b2 = np.array([a/2, a*math.sqrt(3)/6, a*math.sqrt(2/3)])

ri_0 = np.array([np.zeros(3) for i in range(N)])

for l in range(0,n):
    for m in range(0,n):
        for k in range(0,n):
            i = k +m*n + l*n**2
            ri_0[i] = (k - (n-1)/2)*b0 + (m - (n-1)/2)*b1 + (l - (n-1)/2)*b2


with open(f'data/{file_xyz}', 'w') as file:
    file.write(str(N) + '\n\n')
    for atom in range(len(ri_0)):
        file.write(f'Ar {ri_0[atom][0]} {ri_0[atom][1]} {ri_0[atom][2]}\n')

Ekin_i = np.array([np.zeros(3) for i in range(N)])

for i in range(0,N):
    for j in range(0,3):
        lambda_iq = random.uniform(0.000001, 1)
        Ekin_i[i][j] = - 0.5 * 8.31 *10**(-3) * T_0 * np.log(lambda_iq)

Pi_0 = np.array([np.zeros(3) for i in range(N)])
for i in range(0,N):
    for j in range(0,3):
        sign_rand = random.uniform(0,1)
        if sign_rand >= 0.5:
            sign = 1
        else:
            sign = -1
        Pi_0[i][j] = sign * np.sqrt(2 * 40 * Ekin_i[i][j])

Pi_0_df = pd.DataFrame(Pi_0, columns =['x', 'y', 'z'])

Pi_0_prim_df = Pi_0_df.copy()

Pi_0_prim_df['x'] = Pi_0_prim_df['x'] - Pi_0_prim_df['x'].sum()/N
Pi_0_prim_df['y'] = Pi_0_prim_df['y'] - Pi_0_prim_df['y'].sum()/N
Pi_0_prim_df['z'] = Pi_0_prim_df['z'] - Pi_0_prim_df['z'].sum()/N

# ax = Pi_0_prim_df.plot.hist(subplots=True, legend=True, bins = 20, rwidth = 0.9)
# ax[0].set_title('Histogramy pędów')
# ax[2].set_xlabel('Wartość pędu')
# plt.show()

p0nupy = Pi_0_prim_df.to_numpy()
Pi_0_prim = np.array([p0nupy[i] for i in range(N)])

Vp_rij = np.zeros((N,N))

for i in range(0,N):
    for j in range(0,i):
        norm_r_ij = np.linalg.norm((ri_0[i] - ri_0[j]))
        Vp_rij[i][j] = e * ((R/norm_r_ij)**12 - 2*(R/norm_r_ij)**6)
        Vp_rij[j][i] = Vp_rij[i][j]

Vs_ri = np.zeros(N)

for i in range(0,N):
    norm_ri = np.linalg.norm(ri_0[i])
    if norm_ri < L:
        Vs_ri[i] = 0
    else:
        Vs_ri[i] = 0.5 * f * (norm_ri - L)**2

V_0 = 0

for i in range(1, N):
    for j in range(0, i):
        V_0 += Vp_rij[i][j]

V_0 += np.sum(Vs_ri)

F_S_i = np.array([np.zeros(3) for i in range(N)])

for i in range(0,N):
    norm_ri = np.linalg.norm(ri_0[i])
    if norm_ri < L:
        F_S_i[i] = np.zeros(3)
    else:
        F_S_i[i] = f * (L - norm_ri) * ri_0[i]/norm_ri

F_P_i_j = np.array([[np.zeros(3) for x in range(N)] for y in range(N)])

for i in range(0,N):
    for j in range(0,i):
        norm_r_ij = np.linalg.norm((ri_0[i] - ri_0[j]))
        F_P_i_j[i][j] = 12 * e * ((R/norm_r_ij)**12 - (R/norm_r_ij)**6) * (ri_0[i] - ri_0[j])/norm_r_ij**2
        F_P_i_j[j][i] = - F_P_i_j[i][j]

F_i_0 = np.array([np.zeros(3) for i in range(N)])

for i in range(0,N):
    F_i_0[i] = F_S_i[i] + np.sum(F_P_i_j,axis=1)[i]

sum_F_S_i = 0

for i in range(0,N):
    norm_F_S_i = np.linalg.norm((F_S_i[i]))
    sum_F_S_i += norm_F_S_i

P_0 = 1/(4 * np.pi * L**2) * sum_F_S_i

p_i = Pi_0_prim
F_i = F_i_0
r_i = ri_0
V = V_0
P = P_0
t = 0
H = przelicz_hamiltonian(N, M, V, p_i)
T = przelicz_temperatury(N, M, p_i)

del Pi_0_prim, F_i_0, ri_0, V_0, P_0, sum_F_S_i, Vs_ri, Vp_rij, p0nupy, Pi_0_prim_df, Ekin_i, Pi_0, Pi_0_df

with open(f'data/{file_output}', 'w') as file:
    file.write(f't, H, V, T, P\n')
    file.write(f'{"{:.3e}".format(t)}, {"{:.6e}".format(H)},{"{:.3e}".format(V)}, {"{:.3e}".format(T)}, {"{:.3e}".format(P)}\n')

console.print("Wykonywanie symulacji:", style="bold blue")

T_sr = 0
P_sr = 0
H_sr = 0

for i in track(range(S_0 + S_d)):
    
    p_i = przelicz_ped(tau, p_i, F_i)
    r_i = przelicz_polorzenie(M, tau, r_i, p_i)
    F_i, V, P = przelicz_sily(N, e, R, L, f, r_i, F_i, F_S_i, F_P_i_j)
    p_i = przelicz_ped(tau, p_i, F_i)
    H = przelicz_hamiltonian(N, M, V, p_i)
    T = przelicz_temperatury(N, M, p_i)

    t += tau

    if (i+1) % S_out == 0:

        with open(f'data/{file_output}', 'a') as file:
            file.write(f'{"{:.3e}".format(t)}, {"{:.6e}".format(H)},{"{:.3e}".format(V)}, {"{:.3e}".format(T)}, {"{:.3e}".format(P)}\n')

    if (i+1) % S_xyz == 0:
        
        with open(f'data/{file_xyz}', 'a') as file:
            file.write(str(N) + '\n\n')
            for atom in range(N):
                file.write(f'Ar {r_i[atom][0]} {r_i[atom][1]} {r_i[atom][2]}\n')

    if (i+1) >= S_0:
        T_sr += T
        P_sr += P
        H_sr += H

T_sr = T_sr/S_d
P_sr = P_sr/S_d
H_sr = H_sr/S_d

console.print("Wartości uśrednione:", style="bold cyan")
console.print(f"{H_sr = }", style="bold cyan")
console.print(f"{T_sr = }", style="bold cyan")
console.print(f"{P_sr = }", style="bold cyan")