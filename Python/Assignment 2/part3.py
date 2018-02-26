import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from helpers import Option, geo_mean

r = 0.06
sigma = 0.2
T = 1
K = 99
S_0 = 100
N = 365

def BS_Asian(T, N, S_0, K, r, sigma):
    #N = 365
    
    mu = (r - sigma**2/2)*((N+1)/(2*N))
    sigma2 = (N+1)*(2*N+1)*sigma**2/(6*N**2)
    S_0 = S_0*np.exp((mu-r+sigma2/2)*T)
    
    d1 = (np.log(S_0/K) + (r + sigma2/2)*T)/np.sqrt(T*sigma2)
    d2 = d1 - np.sqrt(T*sigma2)
    
    c = S_0*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
    return c

print("The analytic value of the Asian call option is:")
print(BS_Asian(T, N, S_0, K, r, sigma))
    
option = Option('call', S_0, K, r, sigma, T, asian=True)
price, error = option.run_MC(trials=1000, repeats=10)

print("The MC estimated value of the Asian call option is:")
print(price, "+/-", error)
