# python ./wykresy.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from seaborn.rcmod import set_style

df = pd.read_csv('output.txt', sep=",", header=0)

df.columns = ['t', 'H', 'V', 'T', 'P']

df['P'] = df['P']/16.6

#sns.set_style("whitegrid") 
sns.set(style="darkgrid")

fig, axs = plt.subplots(nrows=3)
#axs.ticklabel_format(style='plain', axis='both')
sns.lineplot(x='t', y='H', data=df, ax=axs[0]).set(ylabel='H [kJ/mol]', title = "Wykresy dla n = 6 i T = 0K", xticks = np.arange(0,df['t'].iloc[-1]), xlabel = '', yticks = np.arange(-1251 ,-1250.2,0.2), ylim = (-1251 ,-1250.2))
sns.lineplot(x='t', y='T', data=df, ax=axs[1]).set(ylabel='T [K]', xticks = np.arange(0,df['t'].iloc[-1]), xlabel = '', yticks = np.arange(0 ,6,1), ylim = (0 ,6))
sns.lineplot(x='t', y='P', data=df, ax=axs[2]).set(ylabel='P [atm]', xticks = np.arange(0,df['t'].iloc[-1]), xlabel = '')

plt.xlabel('t [ps]')

plt.show()