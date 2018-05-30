# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:01:52 2018

@author: Felix Farias Fueyo
"""

#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
from ProxyMethods import ProxyMethods

plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

#Loading Data
CDS_data = pd.read_csv('Data/CDS_data.csv')
CDS_hist = pd.read_csv('Data/complete_cds.csv')


''' Pre-Process '''

#Initialize (Parameter Values)
num_index = 3
index_names = ['Region','Sector','AvRating']
SpreadXy = 'Spread5y'
proxy_model = ProxyMethods(num_index, index_names, SpreadXy)
rolling_window = 2 #Months of observations needed for computation of CDS spread rolling volatility

#Match data set with histories to data set with ratings
av_isuers = CDS_data.loc[pd.Series(CDS_data['Ticker']).isin(CDS_hist['Ticker']).values, 'Ticker']
CDS_hist = CDS_hist.set_index('Ticker')
CDS_hist = CDS_hist[pd.Series(CDS_hist.index.values).isin(list(av_isuers)).values]
CDS_hist = CDS_hist.sort_index()

#Prepare data set at date for pre_processing and normalization
non_dates = ['Ticker', 'ShortName', 'Sector', 'Region', 'Country']
CDS_hist[list(set(CDS_hist.columns).difference(non_dates))] = CDS_hist[list(set(CDS_hist.columns).difference(non_dates))]/1000
CDS_hist = CDS_hist[~CDS_hist.index.duplicated()]

#Set index equal to the available issuers
CDS_data = CDS_data.set_index('Ticker')
non_spread_columns = ['Date', 'Timezone', 'ShortName', 'RedCode', 'Tier', 'Ccy',
       'DocClause','Recovery', 'DataRating', 'Sector', 'Region', 'Country',
       'AvRating', 'ImpliedRating']
CDS_data = CDS_data[non_spread_columns]

#Create range of dates in which the proxy methos will be evaluated
Years = np.array([x[0:4] for x in CDS_hist.columns.values])
Months = np.array([x[5:7] for x in CDS_hist.columns.values])
all_dates = CDS_hist.columns.values[np.logical_or(Years == '2018', np.logical_and(Years == '2017', Months >= '04'))]

#Define the models to be tested
num_models = ['InterSection', 'CrossSection', 'CrossSection_v2', 'CDS_vol']

performance_ot_rmse = np.zeros((len(all_dates), len(num_models)))
performance_ot_rmse = pd.DataFrame(performance_ot_rmse)
performance_ot_rmse.columns = num_models
performance_ot_rmse['Date'] = all_dates
performance_ot_rmse.set_index('Date', inplace = True)

for evaluation_dates in all_dates:

    CDS_data = CDS_data.sort_index()
    CDS_data['Spread5y'] = CDS_hist[evaluation_dates] #Initial spreads to be used for modeling and cross-validation.

    #Preprocess the data
    CDS_data = proxy_model.pre_process(CDS_data, low_rating = True, rating_spread_nan = True, currency = True, simple_regions = True, no_gov = True, simple_cat = True, bps_cap = True)

    #Obtain issuers that remain after pre-processing
    av_isuers_ = CDS_data.loc[pd.Series(CDS_data.index.values).isin(CDS_hist.index.values).values].index.values
    CDS_hist = CDS_hist[pd.Series(CDS_hist.index.values).isin(list(av_isuers_)).values]

    #Normalize spread
    recovery_rates = CDS_data['Recovery']
    CDS_data[SpreadXy] = proxy_model.noralize(recovery_rates, CDS_data[SpreadXy])


    ''' Cross-Validation Out-of-Sample '''

    cross_validations = 10 
    sample_size = len(CDS_data)/cross_validations #Cross-validating across isuers.

    performance_rmse = np.zeros((cross_validations, len(num_models)))
    performance_rmse = pd.DataFrame(performance_rmse)
    performance_rmse.columns = num_models
    performance_rmse.index.names = ['Results']

    CDS_vol_bool = True
    for i in range(cross_validations):
        #1. Create N test and sample sets.
        test_set = CDS_data[int((i)*sample_size):int((i+1)*sample_size)]
        CDS_hist_temp = CDS_hist.copy()
    
        #CDS Proxy Pre-Processing
        if CDS_vol_bool == True:
            #Extract x-month volatilities (proxies/non-proxies). 
            CDS_data_temp = CDS_data #Complete set needed for CDS_vol_pre_process() method
            CDS_hist_temp.loc[test_set.index.values, evaluation_dates] = math.nan #For test set, all vols are proxies.
            CDS_hist_temp = proxy_model.CDS_vol_pre_process(CDS_data_temp, CDS_hist_temp, rolling_window, evaluation_dates)
            #Remove buckets for which no issuers have observations at given date
            no_obs_buckets = CDS_hist_temp[CDS_hist_temp.isnull()].reset_index().drop('Ticker', 1).drop_duplicates()[['Region', 'Sector', 'AvRating']]
            CDS_hist_temp = pd.DataFrame(CDS_hist_temp)
            CDS_hist_temp = CDS_hist_temp.reset_index()
            for d in range(len(no_obs_buckets)):
                CDS_hist_temp = CDS_hist_temp[~np.logical_and(np.logical_and(CDS_hist_temp.Region == no_obs_buckets.iloc[d].Region, CDS_hist_temp.Sector == no_obs_buckets.iloc[d].Sector), CDS_hist_temp.AvRating == no_obs_buckets.iloc[d].AvRating)]
                CDS_data_temp = CDS_data_temp[~np.logical_and(np.logical_and(CDS_data_temp.Region == no_obs_buckets.iloc[d].Region, CDS_data_temp.Sector == no_obs_buckets.iloc[d].Sector), CDS_data_temp.AvRating == no_obs_buckets.iloc[d].AvRating)]
                test_set = test_set[~np.logical_and(np.logical_and(test_set.Region == no_obs_buckets.iloc[d].Region, test_set.Sector == no_obs_buckets.iloc[d].Sector), test_set.AvRating == no_obs_buckets.iloc[d].AvRating)]
            
            CDS_hist_temp = CDS_hist_temp.set_index(['Region', 'Sector', 'AvRating', 'Ticker'])
            CDS_hist_temp = CDS_hist_temp.squeeze()
        
        CDS_data_temp = CDS_data_temp.drop(test_set.index, 0)
       
        #2. Apply the methodologies to every possible N-1 data set.
        #Intersection Methodology
        test_set = proxy_model.intersection_method(CDS_data_temp, test_set)
        #Cross-section Methodology (Kandhai et al. 2016)
        test_set = proxy_model.cross_section_method(CDS_data_temp, test_set)
        #Cross-section Methodology v2 (NOMURA's)
        [test_set, parameters] = proxy_model.cross_section_method_v2(CDS_data_temp, test_set)
        #CDS vol Proxy Methodology
        if CDS_vol_bool == True:
            [test_set, parameters_vol] = proxy_model.linear_model(CDS_data_temp, test_set, CDS_hist_temp)
    
    
        #3. Test their accuracy on the 'left-one-out' subsets.
        nans1 = test_set[test_set['InterEstimate'].isnull()].index
        test_set1 = test_set.drop(nans1)
        intersection_rmse = proxy_model.rmse(test_set1[SpreadXy], test_set1['InterEstimate'])
        performance_rmse.loc[performance_rmse.index == i, 'InterSection'] = intersection_rmse
    
        nans2 = test_set[test_set['CrossEstimate'].isnull()].index
        test_set2 = test_set.drop(nans2)
        crosssection_rmse = proxy_model.rmse(test_set2[SpreadXy], test_set2['CrossEstimate'])
        performance_rmse.loc[performance_rmse.index == i, 'CrossSection'] = crosssection_rmse
    
        nans3 = test_set[test_set['CrossEstimate_v2'].isnull()].index
        test_set3 = test_set.drop(nans3)
        crosssection_v2_rmse = proxy_model.rmse(test_set3[SpreadXy], test_set3['CrossEstimate_v2'])
        performance_rmse.loc[performance_rmse.index == i, 'CrossSection_v2'] = crosssection_v2_rmse
        
        nans4 = test_set[test_set['CDS_vol'].isnull()].index
        test_set4 = test_set.drop(nans4)
        CDS_vol_rmse = proxy_model.rmse(test_set4[SpreadXy], test_set4['CDS_vol'])
        performance_rmse.loc[performance_rmse.index == i, 'CDS_vol'] = CDS_vol_rmse
    
    performance_ot_rmse.loc[evaluation_dates,'InterSection'] = np.mean(performance_rmse['InterSection'])
    performance_ot_rmse.loc[evaluation_dates,'CrossSection'] = np.mean(performance_rmse['CrossSection'])
    performance_ot_rmse.loc[evaluation_dates,'CrossSection_v2'] = np.mean(performance_rmse['CrossSection_v2'])
    performance_ot_rmse.loc[evaluation_dates,'CDS_vol'] = np.mean(performance_rmse['CDS_vol'])
    print(evaluation_dates, performance_ot_rmse.loc[evaluation_dates,'InterSection'],
          performance_ot_rmse.loc[evaluation_dates,'CrossSection'],
          performance_ot_rmse.loc[evaluation_dates,'CrossSection_v2'],
          performance_ot_rmse.loc[evaluation_dates,'CDS_vol'])
    
#Save results in CSV format
performance_ot_rmse.to_csv('RMSE_ot.csv')
    
#performance_ot_rmse['YearMonth'] = np.array([x[0:7] for x in performance_ot_rmse.index.values])
lineObjects = plt.plot(performance_ot_rmse, linewidth=3)
plt.xlabel('Date')
plt.ylabel('RMSE')
plt.title('Cross-Validated RMSE Overtime')
plt.legend(iter(lineObjects), ('InterSection', 'CrossSection', 'CDS Vol'))
plt.xticks([0,50,100,150, 200, 250], [performance_ot_rmse.index.values[0],
            performance_ot_rmse.index.values[50],
            performance_ot_rmse.index.values[100],
            performance_ot_rmse.index.values[150],
            performance_ot_rmse.index.values[200],
            performance_ot_rmse.index.values[250]])

    
    
##################                           ######################
##################   Volatility Analysis     ######################
##################                           ######################
    
    
''' Pre-Processing (Volatility) ''' 
    
#Prepare the dataset as done above
CDS_data = pd.read_csv('Data/CDS_data.csv')
CDS_hist = pd.read_csv('Data/complete_cds.csv')

#Initialize (Parameter Values)
num_index = 3
index_names = ['Region','Sector','AvRating']
SpreadXy = 'Spread5y'
proxy_model = ProxyMethods(num_index, index_names, SpreadXy)
rolling_window = 2 #Months of observations needed for computation of CDS spread rolling volatility

#Match data set with histories to data set with ratings
av_isuers = CDS_data.loc[pd.Series(CDS_data['Ticker']).isin(CDS_hist['Ticker']).values, 'Ticker']
CDS_hist = CDS_hist.set_index('Ticker')
CDS_hist = CDS_hist[pd.Series(CDS_hist.index.values).isin(list(av_isuers)).values]
CDS_hist = CDS_hist.sort_index()

#Prepare data set at date for pre_processing and normalization
non_dates = ['Ticker', 'ShortName', 'Sector', 'Region', 'Country']
CDS_hist[list(set(CDS_hist.columns).difference(non_dates))] = CDS_hist[list(set(CDS_hist.columns).difference(non_dates))]/1000
CDS_hist = CDS_hist[~CDS_hist.index.duplicated()]

#Set index equal to the available issuers
CDS_data = CDS_data.set_index('Ticker')
non_spread_columns = ['Date', 'Timezone', 'ShortName', 'RedCode', 'Tier', 'Ccy',
       'DocClause','Recovery', 'DataRating', 'Sector', 'Region', 'Country',
       'AvRating', 'ImpliedRating']
CDS_data = CDS_data[non_spread_columns]

#Create range of dates in which the proxy methos will be evaluated
Years = np.array([x[0:4] for x in CDS_hist.columns.values])
Months = np.array([x[5:7] for x in CDS_hist.columns.values])
all_dates = CDS_hist.columns.values[np.logical_or(Years == '2018', np.logical_and(Years == '2017', Months >= '04'))]


''' Volatility Modelling '''
    
#For the volatility analysis, we need a history of predictions for issuers within selected buckets.
Regions = 'N.Amer'
Sectors = 'Financials'
Ratings = 'BBB'
    
#Test set consists of all issuers within selected buckets at all times. 
test_bucket = CDS_data[np.logical_and(np.logical_and(CDS_data['Region'] == Regions, CDS_data['Sector'] == Sectors), CDS_data['AvRating'] == Ratings)]
test_bucket = test_bucket[['Region', 'Sector', 'AvRating']]

#Obtain proxies for each issuer in the bucket with each model for given date
proxy_intersection = np.zeros((len(all_dates), len(test_bucket)))
proxy_intersection = pd.DataFrame(proxy_intersection)
proxy_intersection.columns = test_bucket.index.values
proxy_intersection['Date'] = all_dates  #Use same dates as for performance analysis analysis.
proxy_intersection.set_index('Date', inplace = True)
proxy_crosssection = proxy_intersection.copy()
proxy_crosssection_v2 = proxy_intersection.copy()
proxy_CDS_vol = proxy_intersection.copy()

for i in range(len(test_bucket)):
    j = 0
    while j < len(all_dates):
        #Pre-process for given date
        CDS_data = CDS_data.sort_index()
        CDS_data['Spread5y'] = CDS_hist[all_dates[j]] #Initial spreads to be used for modeling and cross-validation.

        #Preprocess the data
        CDS_data = proxy_model.pre_process(CDS_data, low_rating = True, rating_spread_nan = True, currency = True, simple_regions = True, no_gov = True, simple_cat = True, bps_cap = True)

        #Obtain issuers that remain after pre-processing
        av_isuers_ = CDS_data.loc[pd.Series(CDS_data.index.values).isin(CDS_hist.index.values).values].index.values
        CDS_hist = CDS_hist[pd.Series(CDS_hist.index.values).isin(list(av_isuers_)).values]

        #Normalize spread
        recovery_rates = CDS_data['Recovery']
        CDS_data[SpreadXy] = proxy_model.noralize(recovery_rates, CDS_data[SpreadXy])
        
        #Determine if issuer satisfied all pre-process requirements
        if test_bucket.index[i] in CDS_data.index.values:
            
            #Get data for test issuer and redifine the CDS historical data.
            test_issuer = pd.DataFrame(CDS_data.loc[test_bucket.index[i]]).transpose()
            CDS_hist_temp = CDS_hist.copy()
        
            #Prepare data for the CDS vol proxy model
            if CDS_vol_bool == True:
                #Extract x-month volatilities (proxies/non-proxies). 
                CDS_data_temp = CDS_data.copy() #Complete set needed for CDS_vol_pre_process() method
                CDS_hist_temp.loc[test_issuer.index.values, all_dates[j]] = math.nan #For test set, all vols are proxies.
                CDS_hist_temp = proxy_model.CDS_vol_pre_process(CDS_data_temp, CDS_hist_temp, rolling_window, all_dates[j])
                #Remove buckets for which no issuers have observations at given date
                no_obs_buckets = CDS_hist_temp[CDS_hist_temp.isnull()].reset_index().drop('Ticker', 1).drop_duplicates()[['Region', 'Sector', 'AvRating']]
                CDS_hist_temp = pd.DataFrame(CDS_hist_temp)
                CDS_hist_temp = CDS_hist_temp.reset_index()
                for d in range(len(no_obs_buckets)):
                    CDS_hist_temp = CDS_hist_temp[~np.logical_and(np.logical_and(CDS_hist_temp.Region == no_obs_buckets.iloc[d].Region, CDS_hist_temp.Sector == no_obs_buckets.iloc[d].Sector), CDS_hist_temp.AvRating == no_obs_buckets.iloc[d].AvRating)]
                    CDS_data_temp = CDS_data_temp[~np.logical_and(np.logical_and(CDS_data_temp.Region == no_obs_buckets.iloc[d].Region, CDS_data_temp.Sector == no_obs_buckets.iloc[d].Sector), CDS_data_temp.AvRating == no_obs_buckets.iloc[d].AvRating)]
                            
                CDS_hist_temp = CDS_hist_temp.set_index(['Region', 'Sector', 'AvRating', 'Ticker'])
                CDS_hist_temp = CDS_hist_temp.squeeze()
        
            #Prepare data for all models
            CDS_data_temp = CDS_data_temp.drop(test_issuer.index.values, 0)
        
            #2. Apply the methodologies to get proxy spread for issuer at given date
            test_issuer = proxy_model.intersection_method(CDS_data_temp, test_issuer)
            #Cross-section Methodology
            test_issuer = proxy_model.cross_section_method(CDS_data_temp, test_issuer)
            #Cross-section (Nomura's)
            [test_issuer, parameters] = proxy_model.cross_section_method_v2(CDS_data_temp, test_issuer)
            #CDS vol Proxy Methodology
            if CDS_vol_bool == True:
                [test_issuer, parameters] = proxy_model.linear_model(CDS_data_temp, test_issuer, CDS_hist_temp)
            
            #3. Store each proxy in the right location
            proxy_intersection.loc[all_dates[j], test_issuer.index.values] = test_issuer['InterEstimate'] 
            proxy_crosssection.loc[all_dates[j], test_issuer.index.values] = test_issuer['CrossEstimate'] 
            proxy_crosssection_v2.loc[all_dates[j], test_issuer.index.values] = test_issuer['CrossEstimate_v2']             
            proxy_CDS_vol.loc[all_dates[j], test_issuer.index.values] = test_issuer['CDS_vol'] 

            print(test_issuer.index.values , test_issuer['InterEstimate'].values, test_issuer['CrossEstimate'].values, 
                  test_issuer['CrossEstimate_v2'].values, test_issuer['CDS_vol'].values)
            
            j= j + 1
        else:
            j = len(all_dates)
            print(test_bucket.index[i], evaluation_dates)

#Save results in CSV format
proxy_intersection.to_csv('proxy_intersection.csv')
proxy_crosssection.to_csv('proxy_crosssection.csv')
proxy_crosssection_v2.to_csv('proxy_crosssection_v2.csv')
proxy_CDS_vol.to_csv('proxy_CDS_vol.csv')

#Remove empty columns
for i in proxy_intersection.columns.values:
    if proxy_intersection[i].mean() == 0:
        proxy_intersection.drop(i, axis = 1, inplace = True)
        proxy_crosssection.drop(i, axis = 1, inplace = True)
        proxy_crosssection_v2.drop(i, axis = 1, inplace = True)
        proxy_CDS_vol.drop(i, axis = 1, inplace = True)
        
#Obtain bucket proxies
bucket_proxy = pd.DataFrame(proxy_intersection.mean(axis = 1))
bucket_proxy.columns = ['InterSection']
bucket_proxy['CrossSection'] = proxy_crosssection.mean(axis = 1)
bucket_proxy['CrossSection v2'] = proxy_crosssection_v2.mean(axis = 1)
bucket_proxy['CDS vol'] = proxy_CDS_vol.mean(axis = 1)

#Plot actual spread values for issuer within bucket.
CDS_bucket = CDS_hist.loc[proxy_intersection.columns.values]
CDS_bucket.drop(['ShortName', 'Sector', 'Region', 'Country'], axis = 1, inplace = True)
CDS_bucket = CDS_bucket.transpose()
CDS_bucket.index.name = 'Date'
CDS_bucket = CDS_bucket.loc[all_dates]
plt.plot(CDS_bucket)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('5y Spread', fontsize = 15)
plt.title('Actual Spreads Overtime', fontsize = 15)
plt.xticks([0,50,100,150, 200, 250], [performance_ot_rmse.index.values[0],
            performance_ot_rmse.index.values[50],
            performance_ot_rmse.index.values[100],
            performance_ot_rmse.index.values[150],
            performance_ot_rmse.index.values[200],
            performance_ot_rmse.index.values[250]])
plt.tick_params(labelsize=12)

#PLot bocket proxies and actual bucket
bucket_proxy['Actual Bucket'] = CDS_bucket.mean(axis = 1)
lineObjects = plt.plot(bucket_proxy, linewidth=3)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('5y Spread', fontsize = 15)
plt.title('Spread Proxies and Actual Spread for Bucket Overtime', fontsize = 15)
plt.legend(iter(lineObjects), ('InterSection', 'CrossSection', 'CrossSection v2', 'CDS Vol', 'Actual Spread'))
plt.xticks([0,50,100,150, 200, 250], [performance_ot_rmse.index.values[0],
            performance_ot_rmse.index.values[50],
            performance_ot_rmse.index.values[100],
            performance_ot_rmse.index.values[150],
            performance_ot_rmse.index.values[200],
            performance_ot_rmse.index.values[250]])
plt.tick_params(labelsize=12)
    
#Plot bucket proxies volatilities    
lineObjects = plt.plot(bucket_proxy.rolling(window= 5,center=False).var(), linewidth=3)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('5y Spread Proxy Volatility', fontsize = 15)
plt.title('Spread Proxies Volatility Overtime', fontsize = 15)
plt.legend(iter(lineObjects), ('InterSection', 'CrossSection', 'CrossSection v2', 'CDS Vol', 'Actual Spread'))
plt.xticks([0,50,100,150, 200, 250], [performance_ot_rmse.index.values[0],
            performance_ot_rmse.index.values[50],
            performance_ot_rmse.index.values[100],
            performance_ot_rmse.index.values[150],
            performance_ot_rmse.index.values[200],
            performance_ot_rmse.index.values[250]])
plt.tick_params(labelsize=12)
    
#PLot individual spreads
#Realization vs Estimates Comparison: Extreme differences even within bucket.. Reasonable to model per bucket? See higher potential by creating buckets based on historical average values (level) and volatility of spreads
plt.plot(proxy_crosssection.index.values, proxy_intersection)
plt.plot(proxy_crosssection.index.values, proxy_crosssection)
plt.plot(proxy_crosssection.index.values, proxy_crosssection_v2)
plt.plot(proxy_crosssection.index.values, proxy_CDS_vol)
plt.xlabel('Date')
plt.ylabel('5y Spread')
plt.title('Spread Proxies Overtime')
plt.xticks([0,50,100,150, 200, 250], [performance_ot_rmse.index.values[0],
            performance_ot_rmse.index.values[50],
            performance_ot_rmse.index.values[100],
            performance_ot_rmse.index.values[150],
            performance_ot_rmse.index.values[200],
            performance_ot_rmse.index.values[250]])   
    




