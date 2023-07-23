# python ./wykresy.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

df = pd.read_csv('output.txt', sep=",", header=0)

df.columns = ['t', 'H', 'V', 'T', 'P']

#sns.set_style("whitegrid") 
sns.set(style="darkgrid")

fig, axs = plt.subplots(nrows=3)

sns.lineplot(x='t', y='H', data=df, ax=axs[0]).set(ylabel='H [kJ/mol]', title = "Wykresy dla n = 5 i T = 1000K", xticks = np.arange(0,df['t'].iloc[-1]), xlabel = '')
sns.lineplot(x='t', y='T', data=df, ax=axs[1]).set(ylabel='T [K]', xticks = np.arange(0,df['t'].iloc[-1]), xlabel = '')
sns.lineplot(x='t', y='P', data=df, ax=axs[2]).set(ylabel='P [Pa]', xticks = np.arange(0,df['t'].iloc[-1]), xlabel = '', yticks = np.arange(0,df['P'].max(), 10))

plt.xlabel('t [ps]')
plt.show()