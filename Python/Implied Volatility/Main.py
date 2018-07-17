# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 12:42:49 2018

@author: felix
@Purpose: Implement and calibrate an implied volatility skew with a polynomial function.
"""
'Useful libraries'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from EurOption import Option
import math
from scipy.optimize import curve_fit
import scipy
plt.rcParams['figure.figsize'] = (10.0, 8.0)

'Load Data'
option_data = pd.read_csv("Data\ABCdata_ext.csv")

'Create Objects of class EurOption with implied volatility'
options = []
option_data['Implied_Volatility'] = math.nan
option_data['Moneyness'] = math.nan
s_0 = 125
r = 0.01
for i in range(0,len(option_data)):
    option_i = Option(option_data.timeToExp[i]/365, option_data.strike[i],
                         option_data.isCallInd[i], option_data.marketPrice[i],
                         s_0, r, False)
    vol_0 = np.sqrt(2/option_i.t*abs(np.log(s_0/option_i.k) + r*option_i.t))
    option_i.implied_volatility(vol_0)
    option_i.compute_moneyness()
    option_data.loc[i, 'Implied_Volatility'] = option_i.sigma_implied
    option_data.loc[i, 'Moneyness'] = option_i.moneyness
    options.append(option_i)

'Calibrate linear function with moneyness as explanatory variable'
#Prepare data set for calibration of semi-parametric model
option_data['Moneyness2'] = option_data['Moneyness']**2
option_data['const'] = 1

#Prepare storage space for results
parameters = ['a_0', 'a_1', 'a_2']
popt = pd.DataFrame(np.zeros((len(option_data.timeToExp.unique()), len(parameters))))
popt.index = option_data.timeToExp.unique()
popt.columns = parameters
perr = pd.DataFrame(np.zeros((len(option_data.timeToExp.unique()), len(parameters))))
perr.index = option_data.timeToExp.unique()
perr.columns = parameters
diagnostic_measures = ['R_sqrd']
diagnostics = pd.DataFrame(np.zeros((len(option_data.timeToExp.unique()), len(diagnostic_measures))))
diagnostics.index = option_data.timeToExp.unique()
diagnostics.columns = diagnostic_measures

option_data.set_index('timeToExp', inplace = True)

#Prepare model
def fn(x, a, b, c):
    return a + b*x[0] + c*x[1]

#Iterate across time to maturity
for i in option_data.index.unique().values:
    X = option_data.loc[i, ['const', 'Moneyness', 'Moneyness2']]
    Y = option_data.loc[i, ['Implied_Volatility']]
    #Alternative 1: Compute ordinary least square estimates FOR EACH EXPIRY OF THE OPTIONS
    #Xt = X.transpose()
    #XtX =  Xt.dot(X)
    #XtX_inv = np.linalg.pinv(XtX)
    #XtX_invXt = XtX_inv.dot(Xt)
    #para_ols = XtX_invXt.dot(Y)
    #Alternative 2: Computing OLS estimates with curve_fit of scipy. Note that although this package focus on curve fitting of non-linear functions, it also works for linear problems.  
    #Define the model to be calibrated. (Although not needed here) Advantage of this method is that it also works for non-linear functions.
    x = scipy.array([X.Moneyness.values, X.Moneyness2.values])
    y = scipy.array(Y.Implied_Volatility.values)
    popt.loc[i], pcov = curve_fit(fn, x, y)
    perr.loc[i] = np.sqrt(np.diag(pcov)) #Standard deviation error of parameters.
    #The comments on Alternative 2 are supported by the result of both approaches being equal.

    #Model fit using Alternative 2:
    #1. Compute residual sum of squares
    residuals = y - fn(x, popt.loc[i, 'a_0'], popt.loc[i, 'a_1'], popt.loc[i, 'a_2'])
    ss_res = np.sum(residuals**2)
    #2. Compute total sum of squares
    ss_tot = np.sum((y - np.mean(y))**2)
    #3. Compute R-squared.
    diagnostics['R_sqrd'] = 1 - (ss_res / ss_tot)

#Compute Vega, Delta and Gamma usin empirical model (obtain semi-parametric implied vol and input into Black-Scholes formulation?)
for i in popt.index.values:
    [option.semi_parametric_iv(popt.loc[i, 'a_0'], popt.loc[i, 'a_1'], popt.loc[i, 'a_2']) for option in options if option.t == i/365]

#Semi-parametric
vol_type = "semipar"
[option.delta(vol_type) for option in options]
[option.vega(vol_type) for option in options]
[option.gamma(vol_type) for option in options]
#Observed
vol_type = "mkt_implied"
[option.delta(vol_type) for option in options]
[option.vega(vol_type) for option in options]
[option.gamma(vol_type) for option in options]

#ToDos: 
#1. Plot volatility smile/skew for each maturity (strike price in x axis)

#Plot the implied volatility term structure. 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
option_data.reset_index(inplace = True)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf([option.moneyness for option in options if option.typeopt], 
                       [option.t*365 for option in options if option.typeopt], 
                       [option.sigma_implied for option in options if option.typeopt], 
                       cmap= cm.jet, linewidth=0.1, vmin=0, vmax=0.5)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title('Implied Volatility Surface');
ax.set_xlabel('Forward-Moneyness')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('IV')
ax.view_init(10, -100)

#Plot the semi-parametric implied volatility term structure. 
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf([option.moneyness for option in options if option.typeopt], 
                       [option.t*365 for option in options if option.typeopt], 
                       [option.sigma_implied_semipar for option in options if option.typeopt], 
                       cmap= cm.jet, linewidth=0.1, vmin=0, vmax=0.5)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title('Semi-Parametric Implied Volatility Surface');
ax.set_xlabel('Forward-Moneyness')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Semi-Parametric IV')
ax.view_init(10, -100)

#Plot the total variance against forward-moneyness to test for calendar arbitrage
[option.compute_total_variance() for option in options]

arb_test3 = pd.DataFrame()
arb_test3_semipar = pd.DataFrame()
sizes = []
st = popt.index.values[0]
arb_test3['Moneyness'] = [option.moneyness for option in options if option.t == st/365  and option.typeopt]
arb_test3.set_index('Moneyness', inplace = True)
arb_test3_semipar['Moneyness'] = [option.moneyness for option in options if option.t == st/365  and option.typeopt]
arb_test3_semipar.set_index('Moneyness', inplace = True)

for i in popt.index.values:
    temp = pd.DataFrame([option.moneyness for option in options if option.t == i/365 and option.typeopt])
    temp["Total_Variance"] = [option.total_var for option in options if option.t == i/365 and option.typeopt]
    temp.columns = ["Moneyness", "Total_Variance_" + str(i)]
    temp.set_index('Moneyness', inplace = True)
    arb_test3 = pd.concat([arb_test3, temp])
    
    temp["Total_Variance_" + str(i)] = [option.total_var_semipar for option in options if option.t == i/365 and option.typeopt]
    arb_test3_semipar = pd.concat([arb_test3_semipar, temp])
    sizes.append([len(temp),i])

plt.grid()
lineObjects = plt.plot(arb_test3)
plt.xlabel('Forward-Moneyness', fontsize = 15)
plt.ylabel('Total Variance', fontsize = 15)
plt.title('Calendar Arbitrage Test', fontsize = 15)
plt.legend(iter(lineObjects), ('30', '60', '90', '120', '150', '180'))
plt.tick_params(labelsize=12)

plt.grid()
lineObjects = plt.plot(arb_test3_semipar)
plt.xlabel('Forward-Moneyness', fontsize = 15)
plt.ylabel('Semi-Parametric Total Variance', fontsize = 15)
plt.title('Calendar Arbitrage Test', fontsize = 15)
plt.legend(iter(lineObjects), ('30', '60', '90', '120', '150', '180'))
plt.tick_params(labelsize=12)

#Plot (for observed and semi parametric in same plot) delta, vega, gamma w.r.t. strike price, maturity, etc. 
#Call
sensitivity = pd.DataFrame([option.delta_mkt for option in options if option.t == 30/365 and option.typeopt])
sensitivity['S.P.-Delta'] = pd.DataFrame([option.delta_semipar for option in options if option.t == 30/365 and option.typeopt])
sensitivity['Moneyness'] = pd.DataFrame([option.moneyness for option in options if option.t == 30/365 and option.typeopt])
sensitivity.columns = ['Delta','S.P.-Delta','Moneyness']
sensitivity.set_index('Moneyness', inplace = True)
plt.grid()
lineObjects = plt.plot(sensitivity)
plt.xlabel('Forward-Moneyness', fontsize = 15)
plt.ylabel('Delta', fontsize = 15)
plt.title('Implied Volatility Delta of Call Option with T = 30', fontsize = 15)
plt.legend(iter(lineObjects), ('Delta', 'S.P.-Delta'))
plt.tick_params(labelsize=12)

sensitivity = pd.DataFrame([option.gamma_mkt for option in options if option.t == 30/365 and option.typeopt])
sensitivity['S.P.-Gamma'] = pd.DataFrame([option.gamma_semipar for option in options if option.t == 30/365 and option.typeopt])
sensitivity['Moneyness'] = pd.DataFrame([option.moneyness for option in options if option.t == 30/365 and option.typeopt])
sensitivity.columns = ['Delta','S.P.-Gamma','Moneyness']
sensitivity.set_index('Moneyness', inplace = True)
plt.grid()
lineObjects = plt.plot(sensitivity)
plt.xlabel('Forward-Moneyness', fontsize = 15)
plt.ylabel('Gamma', fontsize = 15)
plt.title('Implied Volatility Gamma of Call Option with T = 30', fontsize = 15)
plt.legend(iter(lineObjects), ('Gamma', 'S.P.-Gamma'))
plt.tick_params(labelsize=12)

sensitivity = pd.DataFrame([option.vega_mkt for option in options if option.t == 30/365 and option.typeopt])
sensitivity['S.P.-Vea'] = pd.DataFrame([option.vega_semipar for option in options if option.t == 30/365 and option.typeopt])
sensitivity['Moneyness'] = pd.DataFrame([option.moneyness for option in options if option.t == 30/365 and option.typeopt])
sensitivity.columns = ['Vega','S.P.-Vega','Moneyness']
sensitivity.set_index('Moneyness', inplace = True)
plt.grid()
lineObjects = plt.plot(sensitivity)
plt.xlabel('Forward-Moneyness', fontsize = 15)
plt.ylabel('Vega', fontsize = 15)
plt.title('Implied Volatility Vega of Call Option with T = 30', fontsize = 15)
plt.legend(iter(lineObjects), ('Vega', 'S.P.-Vega'))
plt.tick_params(labelsize=12)