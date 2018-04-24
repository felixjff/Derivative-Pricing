import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from helpers import Option

r = 0.06
sigma = 0.2
T = 1
K = 99
S_0 = 100

option = Option('put', S_0, K, r, sigma, T)
price, ci95 = option.run_MC()
print('The estimated price of this option is %f +/- %f'%(price,ci95))

# style for plots
pltstyle = {'elinewidth':1.5, 'fmt':'o', 'capsize':3, 'ecolor':'k', 'mfc':'k', 'mec':'None'}

# different N
N_values = range(100,10101,2000)
prices = []
errors95 = []
with tqdm(len(N_values), disable=False) as pbar:
    option = Option('put', S_0, K, r, sigma, T)
    for N in N_values:
        price, ci95 = option.run_MC(trials=N)
        prices.append(price)
        errors95.append(ci95)
        pbar.update()
               
plt.errorbar(N_values, prices, yerr=errors95, **pltstyle)   
plt.axhline(4.7789, ls='--', c='k', lw=1.)
plt.xlabel('N')
plt.ylabel('Option Price')
plt.show()

# different volatility
prices = []
errors95 = []
sigmas = np.linspace(0,1,10)
with tqdm(len(N_values), disable=False) as pbar:
    for sig in sigmas:
        option = Option('put', S_0, K, r, sig, T)
        price, ci95 = option.run_MC(trials=N)
        prices.append(price)
        errors95.append(ci95)
        pbar.update()     
fig, ax1 = plt.subplots()
ax1.scatter(sigmas, prices, c='k')
ax1.set_xlabel(r'$\sigma$')
ax1.set_ylabel('Option Price', color='k')
ax2 = ax1.twinx()
ax2.scatter(sigmas, errors95, c='r')
ax2.set_ylabel('95% Confidence Interval', color='r')
ax2.tick_params('y', colors='r')
fig.tight_layout()
plt.show()

# different strike price
prices = []
errors95 = []
K_values = range(90,110,2)
with tqdm(len(N_values), disable=False) as pbar:
    for K in K_values:
        option = Option('put', S_0, K, r, sigma, T)
        price, ci95 = option.run_MC(trials=N)
        prices.append(price)
        errors95.append(ci95)
        pbar.update()     
fig, ax1 = plt.subplots()
ax1.scatter(K_values, prices, c='k')
ax1.set_xlabel('K')
ax1.set_ylabel('Option Price', color='k')
ax2 = ax1.twinx()
ax2.scatter(K_values, errors95, c='r')
ax2.set_ylabel('95% Confidence Interval', color='r')
ax2.tick_params('y', colors='r')
fig.tight_layout()
plt.show()
