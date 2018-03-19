import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import *
import time

def PDE(S_0, K, r, sigma, T, t_steps, x_steps):
    dt = T/t_steps
    x_steps = x_steps + x_steps%2
    
    # stock price limits
    x_max = np.log(S_0) + 3*sigma*np.sqrt(T);
    x_min = np.log(S_0) - 3*sigma*np.sqrt(T);
    dx = (x_max - x_min)/x_steps;
    
    X = np.linspace(x_min, x_max, x_steps+1)
    V = np.zeros((t_steps+1, x_steps+1))
    V[:] = np.nan
    
    V[t_steps, :] = [max(np.exp(s)-K,0) for s in X]
    
    # CN
    a_u = (-dt/4)*(sigma**2/dx**2 + (r - 1/2*sigma**2)/dx)
    a_0 = 1 + dt*(sigma**2/(2*dx**2) + r/2)
    a_d = (-dt/4)*(sigma**2/dx**2 - (r - 1/2*sigma**2)/dx)
    
    # FCTS
    #a_u = (-dt/2)*(sigma**2/dx**2 + (r - 1/2*sigma**2)/dx)
    #a_0 = 1 + dt*(sigma**2/(dx**2) + r)
    #a_d = (-dt/2)*(sigma**2/dx**2 - (r - 1/2*sigma**2)/dx)
       
    A = np.zeros((x_steps+1, x_steps+1))
    for i in range(1,x_steps):
        A[i,i-1] = a_d
        A[i,i] = a_0
        A[i,i+1] = a_u
    A[0,0], A[0,1], A[-1,-1], A[-1,-2] = -1, 1, 1, -1
    #A[0,0], A[0,1], A[-1,-1], A[-1,-2] = -1, 1, 1, -1
    
    B = 2*np.identity(x_steps+1) - A  # CN
    #B = np.identity(x_steps+1) # FCTS
    B[0,0], B[0,1], B[-1,-1], B[-1,-2] = 1, -1, 1, -1
    
    for i in range(t_steps, -1, -1):
        V_ = np.dot(B,V[i, :])
        V_[0], V_[-1] = 0, np.exp(X[-1]) - np.exp(X[-2])
        V[i-1, :] = solve(A,V_)
        
    return V[0,int(x_steps/2)], V[0,:], X
 
def black_scholes(option, T, S_0, K, r, sigma):
    '''Calculate the price of a EU call option
       Using Black-Scholes formula'''
    
    d1 = (log(S_0/K) + (r + sigma**2/2)*T)/(sigma*sqrt(T))
    d2 = (log(S_0/K) + (r - sigma**2/2)*T)/(sigma*sqrt(T))
    
    if option == 'call':
        f = S_0*norm.cdf(d1) - exp(-r*T)*K*norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        f = exp(-r*T)*K*norm.cdf(-d2) - S_0*norm.cdf(-d1)
        delta = norm.cdf(d1)-1       
        
    return f, delta
    
# look for best grid sizes

S_0 = 100
T = 1
sigma = 0.3
r = 0.04
K = 110

best = [0,0]
best_cost = 1000000

vals = []
bs_val = black_scholes('call',T, S_0, K, r, sigma)[0]
for t in range(1,500):
    for n in range(1,500):
        c = time.time()
        pde_val = PDE(S_0, K, r, sigma, T, t, n)[0]
        cost = time.time() - c
        err = np.absolute(pde_val - bs_val)
        if err < 0.00001:
            if cost < best_cost:
                best_cost = cost
                best = [t, n]

print(best)
PDE(S_0, K, r, sigma, T, 90, 402)[0] - black_scholes('call',T, S_0, K, r, sigma)[0]


# estimate convergence
p = []
for x in range(500,1000,20):
    h1 = PDE(S_0, K, r, sigma, T, 500, x)[0] - black_scholes('call',T, S_0, K, r, sigma)[0]
    h2 = PDE(S_0, K, r, sigma, T, 500, 2*x)[0] - black_scholes('call',T, S_0, K, r, sigma)[0]
    p.append(log(fabs(h1/h2), 2))
np.mean(p)

# plot delta vs. S
val, f, x = PDE(S_0, K, r, sigma, T, 100, 100)
S = np.exp(x)

deltas = []
for i in range(S.size-1):
    delta = (f[i+1] - f[i])/(S[i+1] - S[i])
    deltas.append(delta)
    
plt.plot(x[:-1],deltas)
plt.xlabel(r'$S$')
plt.ylabel(r'$\Delta$').set_rotation(0)
plt.title('Delta (FCTS)')
plt.show()

