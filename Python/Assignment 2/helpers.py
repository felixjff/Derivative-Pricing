import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Option:
    '''Defines the option contract'''
    
    def __init__(self, option_type, S_0, K, r, sigma, T, digi=False):
        '''Initiate variables'''
        
        self.type = option_type
        self.S_0 = S_0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.digi = digi
        
    def calc_payoff(self, shift=0):
        '''Calculate the payoff of an option'''
        
        S = self.S_0 + shift
        # stock price at maturity (analytic)
        Z = np.random.normal()
        S_T = S*(np.exp(self.r-0.5*self.sigma**2*self.T 
                        + self.sigma*np.sqrt(self.T)*Z))
        
        # Euler scheme
        #S_T = self.S_0
        #steps = 365
        #for t in range(steps):
        #    S_T += S_T*(self.r/steps + self.sigma*np.sqrt(1/steps)*np.random.normal())
        
        # calculate payoff given K
        if self.type == 'call':
            payoff = max(S_T-self.K,0)
        elif self.type == 'put':
            payoff = max(self.K-S_T,0)
        else:
            raise ValueError('Option type not recognized')
        
        if self.digi == True:
            if payoff > 0:
                payoff = 1
            
        return payoff, Z
        
            
    def run_MC(self, shift=0, trials=1000, repeats=1000):
        '''Run Monte-Carlo simulations'''
        
        data = []
        
        for i in range(repeats):
            # discounted value of the average pay-off
            data.append(np.exp(-self.r*self.T)
                        *np.mean([self.calc_payoff(shift=shift)[0] for i in range(trials)]))
       
        # 95% confidence interval
        ci95 = 1.96*np.std(data)/np.sqrt(repeats)
        
        return np.mean(data), ci95
    
    def bump_rev(self, shift, trials=1000, repeats=1000, re_seed=False):
        '''Calculate delta using bump-and-revalue method'''
    
        seed = np.random.randint(0,10)
        np.random.seed(seed)
        value1, error1 = self.run_MC(trials=trials, repeats=repeats)
        
        if re_seed == True:
            np.random.seed(seed)
        value2, error2 = self.run_MC(shift=shift, trials=trials, repeats=repeats)
        
        delta = (value2 - value1)/shift
        total_error = error1 + error2
        
        return delta, total_error
        
    def lh_ratio(self, trials=1000, repeats=1000):
    '''Calculate delta using likelihood ratio estimation'''
                
        S_0 = self.S_0
        
        data = []
        
        for i in range(repeats):
            # calculate the estimator
            sim_data = [self.calc_payoff() for i in range(trials)]
            data.append(np.mean(
                [np.exp(-self.r*self.T)*I*Z/(S_0*self.sigma*np.sqrt(self.T))
                for (I,Z) in sim_data]))
        # 95% confidence interval
        ci95 = 1.96*np.std(data)/np.sqrt(repeats)
        
        return np.mean(data), ci95