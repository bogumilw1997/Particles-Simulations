import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
import rich.traceback
from rich.progress import track
from scipy.optimize import curve_fit
import pandas as pd

def lorentzian(x, gamma, x0):
    return 0.5 * gamma / (np.pi * ((x - x0)**2 + (0.5 * gamma)**2))

rich.traceback.install()
console = Console()

df = pd.read_csv('data/dane.csv')
om_ = df['om/om0']
energy_max_list = df['Max_energy']

popt, pcov = curve_fit(lorentzian, xdata= om_, ydata= energy_max_list)

gamma, x0 = popt[0], popt[1]

gamma_f = "{:.3f}".format(gamma)
x0_f = "{:.2f}".format(x0)

console.print(f"gamma = {gamma_f}")
console.print(f"x0 = {x0_f}")

y_fits = []
x_fits = np.linspace(0.8, 1.2, num = 100, endpoint=True)
x_fits = np.append(x_fits, 1)
x_fits = np.sort(x_fits)

for i in x_fits:
    y_fits.append(lorentzian(i, gamma, x0))
    
sns.lineplot(x = om_, y = energy_max_list, label = 'symulacja')
sns.lineplot(x = x_fits,y = y_fits, label = 'krzywa Lorentza (' + r"$\Gamma$" + '=' + "{:.3f}".format(gamma) + ', ' + r"$x_{0}$" + '=' + "{:.2f}".format(x0) + ')')

plt.ylabel('Max. energia [a.u.]')
plt.xlabel(r"$\omega/\omega_{res}$")

plt.title('Zależność maksymalnej energii od czętsotliwości rezonansowej')
plt.legend()

#plt.savefig('data/fig1.png')
plt.show()