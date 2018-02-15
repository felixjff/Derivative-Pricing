from model import option_price, black_scholes
import numpy as np
import matplotlib.pyplot as plt

# default parameters
N = 50
S_0 = 100
T = 1
sigma = 0.2
r = 0.06
K = 99

price, hedge = option_price('call',T, N, S_0, K, r, sigma)
print('The price of a European call option is: $', round(price,2))
print('The hedge parameter is', round(hedge,2))

# EU call option price for varying N
f = [option_price('call',T, N, S_0, K, r, sigma)[0] for N in range(2,70)]
plt.plot(f)
plt.xlabel('N')
plt.ylabel('f')
plt.grid()
plt.show()

# EU call option price for varying sigma
sigmas = np.linspace(0.01, 0.99, 100)
f = [option_price('call',T, N, S_0, K, r, sigma)[0] for sigma in sigmas]
f2 = [black_scholes('call',T, N, S_0, K, r, sigma)[0] for sigma in sigmas]
plt.plot(sigmas, f, sigmas, f2)
plt.xlabel('sigma')
plt.ylabel('f')
plt.grid()
plt.show()

print('The price of an American call option is: $', round(option_price('call',T, N, S_0, K, r, sigma, AM=True)[0],2))
print('The price of an American put option is: $', round(option_price('put',T, N, S_0, K, r, sigma, AM=True)[0],2))

f_EU = [option_price('call',T, N, S_0, K, r, sigma)[0] for sigma in sigmas]
f_AM = [option_price('call',T, N, S_0, K, r, sigma, AM=True)[0] for sigma in sigmas]
f_diff = [f_AM[i]-f_EU[i] for i in range(len(sigmas))]

plt.plot(sigmas, f_EU, sigmas, f_AM)
plt.xlabel('sigma')
plt.ylabel('f')
plt.grid()
plt.show()