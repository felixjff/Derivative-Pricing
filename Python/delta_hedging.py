import numpy as np
import matplotlib.pyplot as plt
from model import black_scholes

r = 0.06
sigma = 0.2
T = 252
K = 99
S_0 = 100
freq = 7 # hedging interval

# Geometric Brownian Motion
S = [S_0]
for t in range(T):
    S_0 *= np.exp((r-sigma**2/2)*(1/T))*np.exp(sigma*np.random.normal(0,np.sqrt(1/T)))
    S.append(S_0)
    
plt.plot(S)
plt.xlabel('t (day)')
plt.ylabel('value ($)')
plt.title('Stock Movement (Geometric Brownian Motion)')
plt.show()
    
# Calculate option delta and value at time t
deltas = [black_scholes('call', (T-t)/T, S[t], K, r, sigma)[1] for t in range(T-1)]
f = [black_scholes('call', (T-t)/T, S[t], K, r, sigma)[0] for t in range(T-1)]

delta = 0
in_stock = 0
balance = 0
prev_price = S[0]
portfolio_value = []

for t in range(0,T-1,freq):
    
    # difference in option delta
    delta_diff, delta = deltas[t] - delta, deltas[t]
    
    # direction of stock
    stock_dir, prev_price = S[t]/prev_price, S[t]
    
    # buy/sell extra shares
    in_stock = in_stock*stock_dir + delta_diff*S[t]
    
    # balance
    if t == 0:
        balance = -delta_diff*S[t] + f[t] # initial balance
    else:
        balance = -delta_diff*S[t] + np.exp(r*freq/T)*balance     
        
    portfolio_value.append(in_stock + balance)

# sell 1 stock for strike price
balance += K 
in_stock -= S[-1]

# Final balance
print('Final balance': in_stock + balance)

plt.plot(range(T-1),f,label='option price')
plt.plot(range(0,T-1,freq),portfolio_value, label='portfolio value')
plt.xlabel('t (day)')
plt.ylabel('value ($)')
plt.legend(loc='best')
plt.show()

