# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 13:01:35 2018

@author: felix
@Purpose: Option class (package) with implied volatility method.
"""

'Useful libraries'
import numpy as np
import pandas as pd
import math
import scipy as sci
import scipy.stats
import matplotlib.pyplot as plt

class Option:
    #Constructor
    def __init__(self, T, K, Type, Mkt_Price, S_0, r, D):
        #General option information
        self.t = T #Time to maturity
        self.k = K #Strike price
        self.typeopt = Type #Option type: True if Call
        self.mkt_price = Mkt_Price #Market price at T = 0
        self.s_0 = S_0 #Underlying price at T=0
        self.r = r #Interest rate
        self.d = D #If dividend paying
        self.moneyness = math.nan
        #Implied volatilities
        self.sigma_implied = math.nan
        self.sigma_implied_semipar = math.nan
        #Variance
        self.total_var = math.nan
        self.total_var_semipar = math.nan
        #Market implied greeks
        self.delta_mkt = math.nan
        self.gamma_mkt = math.nan
        self.vega_mkt = math.nan
        #Semi-parametric greeks
        self.delta_semipar = math.nan
        self.gamma_semipar = math.nan
        self.vega_semipar = math.nan
        #Arbitrage
    
    #Obtain moneyness of option based on specific definition: ln(k/se^rT) = forward-moneyness
    def compute_moneyness(self):
        if self.typeopt == False:
            self.moneyness = np.log(self.k/self.s_0) - self.r*self.t
        else:
            self.moneyness = np.log(self.s_0/self.k) + self.r*self.t
        return 0
    
    def compute_total_variance(self):
        if self.sigma_implied != math.nan:
            self.total_var = self.sigma_implied*self.t
        else:
            print("To compute total variance of implied volatility, you must compute implied volatility first.")
        
        if self.sigma_implied_semipar != math.nan:
            self.total_var_semipar = self.sigma_implied_semipar*self.t
        else:
            print("To compute total variance of semi-parametric implied volatility, you must compute implied volatility first.")
        
        return 0
        
    #Root finder (Newton-Rampson)
    def implied_volatility(self, vol_0):
        #Difference between market price and theoretical option value.
        def d1(vol_n):
            d1 = (np.log(self.s_0 / self.k) + (self.r + 0.5 * vol_n ** 2) * self.t) / (vol_n * np.sqrt(self.t))
            return d1
        
        def d2(vol_n):
            d2 =  (np.log(self.s_0 / self.k) + (self.r - 0.5 * vol_n ** 2) * self.t) / (vol_n * np.sqrt(self.t))
            return d2
        
        def f(vol_n): 
            if self.typeopt == True:     #Call option      
                f_n = self.s_0 * sci.stats.norm.cdf(d1(vol_n), 0, 1) - self.k * np.exp(-self.r * self.t) * sci.stats.norm.cdf(d2(vol_n), 0, 1) - self.mkt_price
                return f_n
            elif self.typeopt == False: #Put option 
                f_n = self.k * np.exp(-self.r * self.t) * sci.stats.norm.cdf(-d2(vol_n), 0, 1) - self.s_0 * sci.stats.norm.cdf(-d1(vol_n), 0, 1) - self.mkt_price
                return f_n
            
            
        def vega(vol_n):
            #Vega has same functional form for Eur put and call options           
            vega_n = (1 / np.sqrt(2 * np.pi)) * self.s_0 * np.sqrt(self.t) * np.exp(-(sci.stats.norm.cdf(d1(vol_n), 0, 1) ** 2) * 0.5)
            
            return vega_n
        
        #Implement root-finding algorithm
        f_n = f(vol_0)
        vol_new = vol_0
        if f_n != 0:
            while abs(f_n) > 0.001:
                vol_old = vol_new
                vol_new = vol_old - f(vol_old)/vega(vol_old)
                f_n = f(vol_new)
                
        #After convergence, vol_n = implied volatility.
        self.sigma_implied = vol_new
        
        return 0
    
    def semi_parametric_iv(self, a_0, a_1, a_2):
        self.sigma_implied_semipar = a_0 + a_1*self.moneyness + a_2*self.moneyness**2
        return 0
    
    def delta(self, vol_type):
        if vol_type == 'semipar':
            if self.sigma_implied_semipar == math.nan:
                print('Error: To compute Vega using semi-parametric implied volatility you must compute the s.-p. imp. vol. first!')
                return 0
            else:
                d1 = (np.log(self.s_0 / self.k) + (self.r + 0.5 * self.sigma_implied_semipar ** 2) * self.t) / (self.sigma_implied_semipar * np.sqrt(self.t))
                
                if self.typeopt == True: #default is call option
                    delta = sci.stats.norm.cdf(d1, 0, 1)
                else:
                    delta = sci.stats.norm.cdf(d1, 0, 1) - 1
                    
                self.delta_semipar = delta
                return 0
        elif vol_type == 'mkt_implied':
            if self.sigma_implied == math.nan:
                print('Error: To compute Vega using implied volatility you must compute the imp. vol. first!')
                return 0
            else:
                d1 = (np.log(self.s_0 / self.k) + (self.r + 0.5 * self.sigma_implied ** 2) * self.t) / (self.sigma_implied * np.sqrt(self.t))
                
                if self.typeopt == True: #default is call option
                    delta = sci.stats.norm.cdf(d1, 0, 1)
                else:
                    delta = sci.stats.norm.cdf(d1, 0, 1) - 1
                
                self.delta_mkt = delta
            return 0

    
    def gamma(self, vol_type):
        #Gamma has same functional form for Eur put and call options
        if vol_type == 'semipar':
            if self.sigma_implied_semipar == math.nan:
                print('Error: To compute Vega using semi-parametric implied volatility you must compute the s.-p. imp. vol. first!')
                return 0
            else:
                d1 = (np.log(self.s_0 / self.k) + (self.r + 0.5 * self.sigma_implied_semipar ** 2) * self.t) / (self.sigma_implied_semipar * np.sqrt(self.t))
                gamma = sci.stats.norm.pdf(d1) / (self.s_0 * self.sigma_implied_semipar)
                self.gamma_semipar = gamma
                return 0
        elif vol_type == 'mkt_implied':
            if self.sigma_implied == math.nan:
                print('Error: To compute Vega using implied volatility you must compute the imp. vol. first!')
                return 0
            else:
                d1 = (np.log(self.s_0 / self.k) + (self.r + 0.5 * self.sigma_implied ** 2) * self.t) / (self.sigma_implied * np.sqrt(self.t))
                gamma = sci.stats.norm.pdf(d1) / (self.s_0 * self.sigma_implied)
                self.gamma_mkt = gamma
                return 0
               
    def vega(self, vol_type):
        #Vega has same functional form for Eur put and call options
        if vol_type == 'semipar':
            if self.sigma_implied_semipar == math.nan:
                print('Error: To compute Vega using semi-parametric implied volatility you must compute the s.-p. imp. vol. first!')
                return 0
            else:
                d1 = (np.log(self.s_0 / self.k) + (self.r + 0.5 * self.sigma_implied_semipar ** 2) * self.t) / (self.sigma_implied_semipar * np.sqrt(self.t))
                vega = (1 / np.sqrt(2 * np.pi)) * self.s_0 * np.sqrt(self.t) * np.exp(-(sci.stats.norm.cdf(d1, 0, 1) ** 2) * 0.5)
                self.vega_semipar = vega
                return 0
        elif vol_type == 'mkt_implied':
            if self.sigma_implied_semipar == math.nan:
                print('Error: To compute Vega using implied volatility you must compute the imp. vol. first!')
                return 0
            else:
                d1 = (np.log(self.s_0 / self.k) + (self.r + 0.5 * self.sigma_implied ** 2) * self.t) / (self.sigma_implied * np.sqrt(self.t))
                vega = (1 / np.sqrt(2 * np.pi)) * self.s_0 * np.sqrt(self.t) * np.exp(-(sci.stats.norm.cdf(d1, 0, 1) ** 2) * 0.5)
                self.vega_mkt = vega
                return 0
    
    #def arbitrage_test(self):
        #Determine if there are implicity arbitrage opportunities in the dataset
    
        