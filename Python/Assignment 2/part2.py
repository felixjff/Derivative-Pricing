import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from helpers import Option

r = 0.06
sigma = 0.2
T = 1
K = 99
S_0 = 100

shift = 0.01*S_0    # for bump-and revalue
N_values = range(100,10101,2000)  # experiment N-values
N_values = range(100,1001,100)
# style for plots
pltstyle1 = {'elinewidth':1, 'fmt':'o', 'capsize':3, 'ecolor':'k', 'mfc':'k', 'mec':'None'}
pltstyle2 = {'elinewidth':1, 'fmt':'s', 'capsize':3, 'ecolor':'r', 'mfc':'r', 'mec':'None'}

# Bump-and-revalue method for European put

deltas = []
errors = []

option = Option('put', S_0, K, r, sigma, T)

with tqdm(len(N_values), disable=False) as pbar:
  
    for N in N_values:
        delta, error = option.bump_rev(shift=shift, trials=N)
        deltas.append(delta)
        errors.append(error)
        
        pbar.update()

plt.errorbar(N_values, deltas, yerr=errors, **pltstyle)   
plt.axhline(-0.326264, ls='--', c='k', lw=1.)
plt.xlabel('N')
plt.ylabel(r'$\Delta$', rotation='horizontal')
plt.show()


# Bump-and-revalue vs. Likelihood ration method for digital put option

deltas1 = []
deltas2 = []
errors1 = []
errors2 = []

real_val = 0.0180243

option = Option('put', S_0, K, r, sigma, T, digi=True)

with tqdm(len(N_values), disable=False) as pbar:
    for N in N_values:
    
        delta1, error1 = option.bump_rev(shift=shift, trials=N)
        #deltas1.append(np.absolute((delta1-real_val)/real_val))
        deltas1.append(np.absolute(delta1))
        errors1.append(error1)
        #errors1.append(error1/real_val)
        
        delta2, error2 = option.lh_ratio(trials=N)
        #deltas2.append(np.absolute((delta2-real_val)/real_val))
        deltas2.append(np.absolute(delta2))
        errors2.append(error2)
        #errors1.append(error2/real_val)

        pbar.update()
        
#plt.scatter(N_values, deltas1, c='k')
#plt.scatter(N_values, deltas2, c='r')  
plt.errorbar(N_values, deltas1, yerr=errors1, label="bump-and-revalue", **pltstyle1) 
plt.errorbar(N_values, deltas2, yerr=errors2, label="likelihood ratio", **pltstyle2) 
plt.axhline(real_val, ls='--', c='k', lw=1.)
plt.xlabel('N')
plt.ylabel(r'$\Delta$')
plt.grid()
plt.legend(loc='best')
plt.show()