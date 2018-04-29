# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:01:52 2018

@author: Felix Farias Fueyo
"""

#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from ProxyMethods import ProxyMethods

plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

#Loading Data
CDS_data = pd.read_csv('Data/CDS_data.csv')

### Pre-Process ###

#Initialize (Parameter Values)
num_index = 3
index_names = ['Region','Sector','AvRating']
SpreadXy = 'Spread5y'
proxy_model = ProxyMethods(num_index, index_names, SpreadXy)

#Preprocess the data
CDS_data = proxy_model.pre_process(CDS_data, low_rating = True, rating_spread_nan = True, currency = True, simple_regions = True, no_gov = True, simple_cat = True, bps_cap = True)

#Normalize spread
recovery_rates = CDS_data['Recovery']
CDS_data[SpreadXy] = proxy_model.noralize(recovery_rates, CDS_data[SpreadXy])

### Cross-Validation Out-of-Sample ###

num_models = ['InterSection', 'CrossSection', 'Equity', 'EquityVol']
cross_validations = 10
sample_size = len(CDS_data)/cross_validations

performance_rmse = np.zeros((cross_validations, len(num_models)))
performance_rmse = pd.DataFrame(performance_rmse)
performance_rmse.columns = num_models
performance_rmse.index.names = ['Results']
for i in range(cross_validations):
    #1. Create N test and sample sets.
    test_set = CDS_data[int((i)*sample_size):int((i+1)*sample_size)]
    CDS_data_temp = CDS_data.drop(test_set.index)
    
    #2. Apply the methodologies to every possible N-1 data set.
    test_set = proxy_model.intersection_method(CDS_data_temp, test_set)
    test_set = proxy_model.cross_section_method(CDS_data_temp, test_set)

    #3. Test their accuracy on the 'left-one-out' subsets.
    nans1 = test_set[test_set['InterEstimate'].isnull()].index
    test_set1 = test_set.drop(nans1)
    intersection_rmse = proxy_model.rmse(test_set1[SpreadXy], test_set1['InterEstimate'])
    performance_rmse.loc[performance_rmse.index == i, 'InterSection'] = intersection_rmse
    
    nans2 = test_set[test_set['CrossEstimate'].isnull()].index
    test_set2 = test_set.drop(nans2)
    crosssection_rmse = proxy_model.rmse(test_set2[SpreadXy], test_set2['CrossEstimate'])
    performance_rmse.loc[performance_rmse.index == i, 'CrossSection'] = crosssection_rmse
    
inter_rmse = np.mean(performance_rmse['InterSection'])
cross_rmse = np.mean(performance_rmse['CrossSection'])