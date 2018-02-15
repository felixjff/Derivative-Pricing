from math import *

def option_price(option, T, N, S_0, K, r, sigma, AM=False):
    '''Calculate the price of an option derivative'''
    
    u = exp(sigma*sqrt(T/N))
    d = exp(-sigma*sqrt(T/N))
    a = exp(r*T/N)
    p = (a-d)/(u-d)
    
    # stock prices at maturity
    S_T = [S_0*pow(u,j)*pow(d,N-j) for j in range(N+1)]

    # multiplier for call/put
    m = 1 if option == 'call' else -1
    
    # option values at maturity
    f = [max(0,m*(s-K)) for s in S_T]

    # backward iteration
    for i in range(N-1,-1,-1):
        
        # stock prices at i
        S = [S_0*pow(u,j)*pow(d,i-j) for j in range(i+1)]
        
        # excersise values at i
        f_ex = [max(0,m*(s-K)) for s in S]
        
        # model values at i
        f_model = [exp(-r*T/N)*(p*f[j] + (1-p)*f[j+1]) for j in range(i+1)]
        
        # choose max if option is American
        f = [max(f_ex[j],f_model[j]) for j  in range(i+1)] if AM == True else f_model
    
        # calculate hedge parameter
        if i == 1:
            delta = (f[1] - f[0])/(S[1] - S[0])
    
    return f[0], delta
    
def black_scholes(option, T, N, S_0, K, r, sigma):
    '''Calculate the price of a EU call option
       Using Black-Scholes formula'''
    
    d1 = (log(S_0/K) + (r + sigma**2/2)*T)/(sigma*sqrt(T))
    d2 = (log(S_0/K) + (r - sigma**2/2)*T)/(sigma*sqrt(T))
    
    if option == 'call':
        f = S_0*norm.cdf(d1) - exp(-r*T)*K*norm.cdf(d2)
    else:
        f = exp(-r*T)*K*norm.cdf(-d2) - S_0*norm.cdf(-d1)
        
    return f, norm.cdf(d2)
    