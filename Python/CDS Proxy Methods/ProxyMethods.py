# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:08:50 2018

@author: Felix Farias Fueyo
"""
#Loading libraries 
import numpy as np 
import pandas as pd
import math

class ProxyMethods:
    
    def __init__(self, num_index, index_names, SpreadXy):
        #number of indeces used to divide data set/create buckets (numeric)
        self.num_index = num_index
        #name of indeces used to create buckets (vector)
        self.index_names = index_names
        #spread maturity, where spreadXy = the X year spread. (string 'Spread{int}y')
        self.spreadXy = SpreadXy 
        
    #Prepare the data set for proxy modeling
    def pre_process(self, CDS_data, low_rating, rating_spread_nan, currency, simple_regions, no_gov, simple_cat, bps_cap):
        CDS_data = pd.DataFrame(CDS_data)
        
        #Remove CCC and D rated credits. Their spreads considered outliers.
        if low_rating == True:
           CDS_data = CDS_data.loc[np.logical_and(CDS_data['AvRating'] != 'CCC', CDS_data['AvRating'] != 'D')]
                   
        #Remove observations with no rating
        if rating_spread_nan == True:
            rating_nans = CDS_data[CDS_data['AvRating'].isnull()].index
            CDS_data = CDS_data.drop(rating_nans)
            spread_nans = CDS_data[CDS_data[self.spreadXy].isnull()].index
            CDS_data = CDS_data.drop(spread_nans)
        
        #Retain quotes in currencies according to geographic regions
        if currency == True:
            eu_countries = ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 'Lithuania','Luxembourg','The Netherlands','Norway','Poland','Portugal','Spain','Sweden','Switzerland','United Kingdom']
            for i in eu_countries:
                CDS_data.loc[CDS_data['Country'] == i] = CDS_data.loc[np.logical_and(CDS_data['Country'] == i, CDS_data['Ccy'] == 'EUR')]
        
        #Redifine regions which are not Asia, Europe or North America as 'Other'
        if simple_regions == True:
            regions = list(CDS_data['Region'].unique())
            regions.remove('Asia')
            regions.remove('N.Amer')
            regions.remove('Europe')
            for i in regions:
                CDS_data.loc[CDS_data['Region'] == i, 'Region'] = 'Other'
        
        #Remove Sovereign quotes
        if no_gov == True:
            CDS_data = CDS_data.loc[CDS_data['Sector'] != 'Government']
            
        #Redifine the sector categories to a smaller set
        if simple_cat == True:
            cyclical = ['Basic Materials', 'Consumer Services', 'Energy', 'Technology', 'Telecommunications Services', 'Industrials']
            for i in cyclical:
                CDS_data.loc[CDS_data['Sector'] == i, 'Sector'] = 'Cyclical'
            
            non_cyclical = ['Consumer Goods', 'Healthcare', 'Utilities']
            for i in non_cyclical:
                CDS_data.loc[CDS_data['Sector'] == i, 'Sector'] = 'Non Cyclical'
        
        #Cap the max. spread to 1000 bps.
        if bps_cap == True:
            CDS_data = CDS_data.loc[CDS_data[self.spreadXy] < 0.10]
            
        return CDS_data
        
    #normalize using simplified approach. Extension possible for hazard rate approach
    def noralize(self, RR, CDS_spread):
        norm_RR = CDS_spread*0.6/(1-RR)
        
        return norm_RR
    
    #Intersection method (for pre-processed data)
    def intersection_method(self, CDS_data, test_set):
        #Obtain all different categories of highest hierarchy
        h_cat = test_set[self.index_names[0]].unique()
        
        #Obtain proxy spread for each test observation by looping over the
        for i in h_cat:
            #For the given region determine what sectors are in test_set
            h2_cat = []
            h2_cat = test_set.loc[test_set[self.index_names[0]] == i, self.index_names[1]].unique()
            if len(h2_cat) != 0:
                #Spread for every possible h1, h2 combination
                for j in h2_cat:
                    h3_cat = []
                    h3_cat = test_set.loc[np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), self.index_names[2]].unique()
                    if len(h3_cat) != 0:
                        for m in h3_cat:
                            #Get intersection estimate for test obs. with region = i, ...
                            s_hat = CDS_data.loc[np.logical_and(np.logical_and(CDS_data[self.index_names[0]] == i, CDS_data[self.index_names[1]] == j), CDS_data[self.index_names[2]] == m), self.spreadXy]
                            #Assign estimate to test obs. with region = i, ...
                            test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'InterEstimate'] = np.nanmean(s_hat)
                            
        return test_set
    
    #Cross-section method based on Kandhai, Hofer, Sourabh "Liquidity risk in derivatives valuation: an improved credit proxy method" 
    def cross_section_method(self, CDS_data, test_set):
        test_set = test_set.assign( CrossEstimate = pd.Series(np.zeros(len(test_set))))
        #Log-transformation of spread
        CDS_data[self.spreadXy] = np.log(CDS_data[self.spreadXy])
        
        #Intercept based: Europe > Financials > A
        b_0 = np.mean(CDS_data.loc[np.logical_and(np.logical_and(CDS_data[self.index_names[0]] == 'Europe', CDS_data[self.index_names[1]] == 'Financials'), CDS_data[self.index_names[2]] == 'A'), self.spreadXy])
        test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == 'Europe', test_set[self.index_names[1]] == 'Financials'), test_set[self.index_names[2]] == 'A'), 'CrossEstimate'] = b_0
        
        h_cat = test_set[self.index_names[0]].unique()

        for i in h_cat:
            
            #If not in intercept, compute difference from intercept value to current value
            if i != 'Europe':
                #Step to other region
                step_region = np.mean(CDS_data.loc[np.logical_and(np.logical_and(CDS_data[self.index_names[0]] == i, CDS_data[self.index_names[1]] == 'Financials'), CDS_data[self.index_names[2]] == 'A'), self.spreadXy])
                #Region factor: from Europe to current region
                b_1 = step_region - b_0
                
            h2_cat = []
            h2_cat = test_set.loc[test_set[self.index_names[0]] == i, self.index_names[1]].unique()
            if len(h2_cat) != 0:
                for j in h2_cat: 
                    
                    #If not in intercept, compute difference from intercept value to current value
                    if j != 'Financials':
                        #Step to other sector
                        step_sector = np.mean(CDS_data.loc[np.logical_and(np.logical_and(CDS_data[self.index_names[0]] == 'Europe', CDS_data[self.index_names[1]] == j), CDS_data[self.index_names[2]] == 'A'), self.spreadXy])
                        #Sector factor: from Financials to current sector
                        b_2 = step_sector - b_0
                    
                    h3_cat = []
                    h3_cat = test_set.loc[np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), self.index_names[2]].unique()
                    if len(h3_cat) != 0:
                        for m in h3_cat:
                            
                            #If not the intercept, compute difference from intercept value to current value
                            if m != 'A' :
                                #Step to other rating
                                step_rating = np.mean(CDS_data.loc[np.logical_and(np.logical_and(CDS_data[self.index_names[0]] == 'Europe', CDS_data[self.index_names[1]] == 'Financials'), CDS_data[self.index_names[2]] == m), self.spreadXy])
                                #Sector factor: from A to current rating
                                b_3 = step_rating - b_0
                            #Compute estimate
                            if i != 'Europe' and j != 'Financials' and m != 'A' :
                                test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'CrossEstimate'] = b_1 + b_2 + b_3 + b_0
                            elif i == 'Europe' and j == 'Financials' and m != 'A':
                                test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'CrossEstimate'] = b_3 + b_0 
                            elif i == 'Europe' and j != 'Financials' and m == 'A':
                                test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'CrossEstimate'] = b_2 + b_0 
                            elif i != 'Europe' and j == 'Financials' and m == 'A':
                                test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'CrossEstimate'] = b_1 + b_0 
                            elif i == 'Europe' and j != 'Financials' and m != 'A':
                                test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'CrossEstimate'] = b_2 + b_3 + b_0 
                            elif i != 'Europe' and j == 'Financials' and m != 'A':
                                test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'CrossEstimate'] = b_1 + b_3 + b_0 
                            elif i != 'Europe' and j != 'Financials' and m == 'A':
                                test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'CrossEstimate'] = b_1 + b_2 + b_0 
                            elif i == 'Europe' and j == 'Financials' and m == 'A':
                                test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'CrossEstimate'] = b_0 

        test_set['CrossEstimate'] = np.exp(test_set['CrossEstimate'])
        
        return test_set
    
    #Compute volatilities for the CDS spreads in the training set. Vols of test set are computed using CDS vol proxy
    def CDS_vol_pre_process(self, CDS_data, CDS_hist, rolling_window, evaluation_dates):
        non_dates = ['ShortName', 'Sector', 'Region', 'Country']

        #Compute rolling volatilities for each entity. 
        CDS_hist_temp = CDS_hist[list(set(CDS_hist.columns).difference(non_dates))]
        CDS_hist_temp = CDS_hist_temp.transpose()
        CDS_hist_temp = CDS_hist_temp.sort_index(axis =0)
        CDS_hist_temp = CDS_hist_temp.rolling(window= 30*rolling_window,center=False).var()
        CDS_hist_temp = CDS_hist_temp.transpose()
        #Add indeces to form bucket vols from
        CDS_hist_temp['Region'] = CDS_data['Region'].values
        CDS_hist_temp['Sector'] = CDS_data['Sector'].values
        #Extract AvRating from CDS_data
        CDS_hist_temp['AvRating'] = np.zeros(len(CDS_hist_temp))
        for i in CDS_hist_temp.index.values:
            CDS_hist_temp.loc[i, 'AvRating'] = CDS_data.loc[i, 'AvRating']

        #Use Cross-Section methodology to imput missing values for each date
        CDS_hist_temp['Ticker'] = CDS_hist_temp.index.values
        CDS_hist_temp = CDS_hist_temp.set_index(['Region', 'Sector', 'AvRating', 'Ticker'])
        CDS_hist_temp = np.log(CDS_hist_temp)
        
        #Intercept value for each date.
        d_0 =  CDS_hist_temp.loc['Europe', 'Financials', 'A'].mean()
        
        #Use the CDS vol proxy to fill-in missing values
        h_cat = CDS_hist_temp.index.get_level_values('Region').unique().values
        #For each day we will input missing values using the cross-section method
        for r in h_cat: 
            #If not in intercept, compute difference from intercept value to current value
            if r != 'Europe':
                #Step to other region
                step_region = CDS_hist_temp.loc[r, 'Financials', 'A'].mean()
                #Region factor: from Europe to current region
                d_1 = step_region - d_0
                
            h_cat1 = CDS_hist_temp.loc[r].index.get_level_values('Sector').unique().values 
            for s in h_cat1:
                #If not in intercept, compute difference from intercept value to current value
                if s != 'Financials':
                    #Step to other sector
                    step_sector = CDS_hist_temp.loc['Europe', s, 'A'].mean()
                    #Sector factor: from Financials to current sector
                    d_2 = step_sector - d_0
                    
                h_cat2 = CDS_hist_temp.loc[r,s].index.get_level_values('AvRating').unique().values
                for ar in h_cat2:
                    #If not the intercept, compute difference from intercept value to current value
                    if ar != 'A' :
                        if any(CDS_hist_temp.loc['Europe', 'Financials'].index.get_level_values('AvRating').unique() == ar):
                            step_rating = CDS_hist_temp.loc['Europe', 'Financials', ar].mean()
                            d_3 = step_rating - d_0
                        else:
                            d_3 = CDS_hist_temp.loc[r, s, ar].mean() - d_0
                        
                    #Imput missing values for each date where bucket value is available but not the individual value
                    if r != 'Europe' and s != 'Financials' and ar != 'A' :
                        proxy_series = d_1 + d_2 + d_3 + d_0
                        for d in [evaluation_dates]:
                            #Check if there is missing value and proxy is avilable at bucket level
                            if CDS_hist_temp.loc[r, s, ar][d].isnull().sum() > 0 and ~np.isnan(proxy_series[d]):
                                #Determine which isuers have a missing value for given date
                                d_missing = CDS_hist_temp.loc[(r, s, ar), d][CDS_hist_temp.loc[r, s, ar][d].isnull()].index.values
                                #Imput missing values
                                for nan in d_missing:
                                    CDS_hist_temp.loc[(r, s, ar, nan), d] = proxy_series[d]
                            
                    elif r == 'Europe' and s == 'Financials' and ar != 'A':
                         proxy_series = d_3 + d_0
                         for d in [evaluation_dates]:
                             if CDS_hist_temp.loc[r, s, ar][d].isnull().sum() > 0 and ~np.isnan(proxy_series[d]):
                                 d_missing = CDS_hist_temp.loc[(r, s, ar), d][CDS_hist_temp.loc[r, s, ar][d].isnull()].index.values
                                 for nan in d_missing:
                                    CDS_hist_temp.loc[(r, s, ar, nan), d] = proxy_series[d]
                         
                    elif r == 'Europe' and s != 'Financials' and ar == 'A':
                        proxy_series = d_2 + d_0
                        for d in [evaluation_dates]:
                             if CDS_hist_temp.loc[r, s, ar][d].isnull().sum() > 0 and ~np.isnan(proxy_series[d]):
                                 d_missing = CDS_hist_temp.loc[(r, s, ar), d][CDS_hist_temp.loc[r, s, ar][d].isnull()].index.values
                                 for nan in d_missing:
                                    CDS_hist_temp.loc[(r, s, ar, nan), d] = proxy_series[d]
                                    
                    elif r != 'Europe' and s == 'Financials' and ar == 'A':
                        proxy_series = d_1 + d_0
                        for d in [evaluation_dates]:
                             if CDS_hist_temp.loc[r, s, ar][d].isnull().sum() > 0 and ~np.isnan(proxy_series[d]):
                                 d_missing = CDS_hist_temp.loc[(r, s, ar), d][CDS_hist_temp.loc[r, s, ar][d].isnull()].index.values
                                 for nan in d_missing:
                                    CDS_hist_temp.loc[(r, s, ar, nan), d] = proxy_series[d]
                                    
                    elif r == 'Europe' and s != 'Financials' and ar != 'A':
                        proxy_series = d_2 + d_3 + d_0
                        for d in [evaluation_dates]:
                             if CDS_hist_temp.loc[r, s, ar][d].isnull().sum() > 0 and ~np.isnan(proxy_series[d]):
                                 d_missing = CDS_hist_temp.loc[(r, s, ar), d][CDS_hist_temp.loc[r, s, ar][d].isnull()].index.values
                                 for nan in d_missing:
                                    CDS_hist_temp.loc[(r, s, ar, nan), d] = proxy_series[d]
                                    
                    elif r != 'Europe' and s == 'Financials' and ar != 'A':
                        proxy_series = d_1 + d_3 + d_0
                        for d in [evaluation_dates]:
                             if CDS_hist_temp.loc[r, s, ar][d].isnull().sum() > 0 and ~np.isnan(proxy_series[d]):
                                 d_missing = CDS_hist_temp.loc[(r, s, ar), d][CDS_hist_temp.loc[r, s, ar][d].isnull()].index.values
                                 for nan in d_missing:
                                    CDS_hist_temp.loc[(r, s, ar, nan), d] = proxy_series[d] 
                                    
                    elif r != 'Europe' and s != 'Financials' and ar == 'A':
                        proxy_series = d_1 + d_2 + d_0
                        for d in [evaluation_dates]:
                             if CDS_hist_temp.loc[r, s, ar][d].isnull().sum() > 0 and ~np.isnan(proxy_series[d]):
                                 d_missing = CDS_hist_temp.loc[(r, s, ar), d][CDS_hist_temp.loc[r, s, ar][d].isnull()].index.values
                                 for nan in d_missing:
                                    CDS_hist_temp.loc[(r, s, ar, nan), d] = proxy_series[d]
                    elif r == 'Europe' and s == 'Financials' and ar == 'A':
                        proxy_series = d_0
                        for d in [evaluation_dates]:
                             if CDS_hist_temp.loc[r, s, ar][d].isnull().sum() > 0 and ~np.isnan(proxy_series[d]):
                                 d_missing = CDS_hist_temp.loc[(r, s, ar), d][CDS_hist_temp.loc[r, s, ar][d].isnull()].index.values
                                 for nan in d_missing:
                                    CDS_hist_temp.loc[(r, s, ar, nan), d] = proxy_series[d]
        
        #Select only the columns that are relevant for the analysis in linear_model
        CDS_hist_temp = CDS_hist_temp[evaluation_dates]
                          
        return CDS_hist_temp
    
    #Cross-section method based on NOMURA's "A cross-section method across CVA"
    def cross_section_method_v2(self, CDS_data, test_set):
        #Define all coefficients of the model
        b = ['Intercept']     #Intercept corresponds to ['Europe', 'Financials', 'A']
        b.extend(list(CDS_data.Region.unique()))
        b.extend(CDS_data.Sector.unique())
        b.extend(CDS_data.AvRating.unique())
        
        beta = []
        beta.extend(b)
        
        regions = CDS_data.Region.unique()
        sectors = CDS_data.Sector.unique()
        ratings = CDS_data.AvRating.unique()
        
        CDS_data_ = CDS_data.copy()
        CDS_data_.reset_index(inplace = True)
        CDS_data_.set_index(['Region', 'Sector', 'AvRating', 'Ticker'], inplace = True)
        CDS_data_ = CDS_data_.squeeze()
        
        #Create X matrix
        X = pd.DataFrame(CDS_data_)
        X = X.drop(list(X.columns), axis = 1 )
        for i in beta:
            X[i] = pd.Series(np.zeros(len(X))).values
            if i == 'Intercept':
                #Global factor
                X[i] = X[i] + 1
            #Only dummy vectors for non-vol variables
            elif any(regions == i):
                #Factor for the region of the obligor
                X.loc[(i), i] = 1
            elif any(sectors == i):
                #Factor for the sector of the obligor
                X.loc[(list(regions), i), i] = 1
            elif any(ratings == i):
                #Factor for the rating of the obligor
                X.loc[(list(regions), list(sectors), i), i] = 1
        
        #Given the matrix of factors X, obtain the OLS estimate for beta = (X'X)^-1 X'y
        X_ = X.reset_index(drop = True)
        X_ = X_.transpose()
        X_1 = X_.dot(X.reset_index(drop = True))
        X_inv = pd.DataFrame(np.linalg.pinv(X_1.values), X_1.columns, X_1.index)
        X_2 = X_inv.dot(X_)
        beta_est = X_2.dot(CDS_data['Spread5y'].values)
        
        #Compute the estimates for test set
        test_set['CrossEstimate_v2'] = math.nan
        for r in regions:
            h_cat1 = test_set.loc[test_set['Region'] == r, 'Sector'].unique()
            for s in h_cat1:
                h_cat2 = test_set.loc[np.logical_and(test_set['Region'] == r,test_set['Sector'] == s), 'AvRating'].unique()
                for ar in h_cat2:
                    proxy_series = beta_est['Intercept'] + beta_est[r] + beta_est[s] + beta_est[ar] #Global factor + region factor + sector factor + rating factor
                    
                    test_set.loc[np.logical_and(test_set['Region'] == r,
                                 np.logical_and(test_set['Sector'] == s, 
                                 test_set['AvRating'] == ar)), 'CrossEstimate_v2'] = proxy_series  
        
        test_set['CrossEstimate_v2'] = np.exp(test_set['CrossEstimate_v2'])
        return [test_set, beta_est]
    
    #Extended cross-section method with explanatory variables: OLS estimates: beta = (X'X)^-1 X'y
    def linear_model(self, CDS_data, test_set, CDS_hist_temp):
        #Define all coefficients of the model
        b = ['Intercept']     #Intercept corresponds to ['Europe', 'Financials', 'A']
        b.extend(list(CDS_hist_temp.index.get_level_values('Region').unique().values))
        b.remove('Europe')
        b.extend(CDS_hist_temp.index.get_level_values('Sector').unique().values)
        b.remove('Financials')
        b.extend(CDS_hist_temp.index.get_level_values('AvRating').unique().values)
        b.remove('A')
        v = ['Intercept vol']    #Intercept corresponds to ['Europe', 'Financials', 'A']
        regions = CDS_hist_temp.index.get_level_values('Region').unique().values
        v.extend(regions + ' vol')
        v.remove('Europe vol')
        sectors = CDS_hist_temp.index.get_level_values('Sector').unique().values
        v.extend(sectors + ' vol')
        v.remove('Financials vol')
        ratings = CDS_hist_temp.index.get_level_values('AvRating').unique().values
        v.extend(ratings + ' vol')
        v.remove('A vol')
        beta = []
        beta.extend(b)
        beta.extend(v)
        
        CDS_hist_temp_in = CDS_hist_temp.reset_index(['Region', 'Sector', 'AvRating'])
        CDS_hist_temp_in = CDS_hist_temp_in.loc[CDS_data.index]
        CDS_hist_temp_in.reset_index(inplace = True)
        CDS_hist_temp_in.set_index(['Region', 'Sector', 'AvRating', 'Ticker'], inplace = True)
        CDS_hist_temp_in = CDS_hist_temp_in.squeeze()
        
        
        #Create X matrix
        X = pd.DataFrame(CDS_hist_temp_in)
        X = X.drop(list(X.columns), axis = 1 )
        for i in beta:
            X[i] = pd.Series(np.zeros(len(X))).values
            if i == 'Intercept':
                X[i] = X[i] + 1
            elif i == 'Intercept vol':
                X[i] = X[i] + CDS_hist_temp_in.values
            elif i[len(i)-3:len(i)] == 'vol': #apply different methodolgy for vol
                var_dummy = i[:len(i)-4]
                if any(regions == i[:len(i)-4]):
                    X.loc[(var_dummy), i] = CDS_hist_temp_in.loc[(var_dummy)].values
                elif any(sectors == i[:len(i)-4]):
                    X.loc[(list(regions), var_dummy), i] = X.loc[(list(regions), var_dummy), i] + CDS_hist_temp_in.loc[(list(regions), var_dummy)].values
                elif any(ratings == i[:len(i)-4]):
                    X.loc[(list(regions), list(sectors), var_dummy), i] = X.loc[(list(regions), list(sectors), var_dummy), i] + CDS_hist_temp_in.loc[(list(regions), list(sectors), var_dummy)].values
            else:  #Only dummy vectors for non-vol variables
                if any(regions == i):
                    X.loc[(i), i] = 1
                elif any(sectors == i):
                    X.loc[(list(regions), i), i] = 1
                elif any(ratings == i):
                    X.loc[(list(regions), list(sectors), i), i] = 1
            
        #Given the matrix of explanatory variables X, obtain the OLS estimate for beta = (X'X)^-1 X'y
        X_ = X.reset_index(drop = True)
        X_ = X_.transpose()
        X_1 = X_.dot(X.reset_index(drop = True))
        X_inv = pd.DataFrame(np.linalg.pinv(X_1.values), X_1.columns, X_1.index)
        X_2 = X_inv.dot(X_)
        beta_est = X_2.dot(CDS_data['Spread5y'].values)
        
        #Compute the estimates for test set
        test_set['CDS_vol'] = math.nan
        for r in regions:
            h_cat1 = test_set.loc[test_set['Region'] == r, 'Sector'].unique()
            for s in h_cat1:
                h_cat2 = test_set.loc[np.logical_and(test_set['Region'] == r,test_set['Sector'] == s), 'AvRating'].unique()
                for ar in h_cat2:
                    if r == 'Europe' and s == 'Financials' and ar == 'A' :
                        test_issuers = test_set.loc[np.logical_and(test_set['Region'] == r, np.logical_and(test_set['Sector'] == s, test_set['AvRating'] == ar)), 'CDS_vol'].index.values
                        proxy_series = beta_est['Intercept'] + CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est['Intercept vol']
                        
                        test_set.loc[np.logical_and(test_set['Region'] == r,
                                                    np.logical_and(test_set['Sector'] == s, 
                                                                   test_set['AvRating'] == ar)), 'CDS_vol'] = proxy_series.values  #Non vol component + vol components
                    elif r != 'Europe' and s != 'Financials' and ar != 'A' :
                        test_issuers = test_set.loc[np.logical_and(test_set['Region'] == r, np.logical_and(test_set['Sector'] == s, test_set['AvRating'] == ar)), 'CDS_vol'].index.values
                        proxy_series = (beta_est['Intercept'] + beta_est[r] + beta_est[s] + beta_est[ar] +  
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est['Intercept vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[r + ' vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[s + ' vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[ar + ' vol']) # all vol/non-vol components
                                        
                        test_set.loc[np.logical_and(test_set['Region'] == r,
                                                    np.logical_and(test_set['Sector'] == s, 
                                                                   test_set['AvRating'] == ar)), 'CDS_vol'] = proxy_series.values  #Non vol component + vol components
                    elif r == 'Europe' and s == 'Financials' and ar != 'A':
                        test_issuers = test_set.loc[np.logical_and(test_set['Region'] == r, np.logical_and(test_set['Sector'] == s, test_set['AvRating'] == ar)), 'CDS_vol'].index.values
                        proxy_series = (beta_est['Intercept'] + beta_est[ar] +  
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est['Intercept vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[ar + ' vol']) # all vol/non-vol components
                                        
                        test_set.loc[np.logical_and(test_set['Region'] == r,
                                                    np.logical_and(test_set['Sector'] == s, 
                                                                   test_set['AvRating'] == ar)), 'CDS_vol'] = proxy_series.values  #Non vol component + vol components
                    elif r == 'Europe' and s != 'Financials' and ar == 'A':
                        test_issuers = test_set.loc[np.logical_and(test_set['Region'] == r, np.logical_and(test_set['Sector'] == s, test_set['AvRating'] == ar)), 'CDS_vol'].index.values
                        proxy_series = (beta_est['Intercept'] + beta_est[s] + 
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est['Intercept vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[s + ' vol']) # all vol components
                                        
                        test_set.loc[np.logical_and(test_set['Region'] == r,
                                                    np.logical_and(test_set['Sector'] == s, 
                                                                   test_set['AvRating'] == ar)), 'CDS_vol'] = proxy_series.values  #Non vol component + vol components
                    elif r != 'Europe' and s == 'Financials' and ar == 'A':
                        test_issuers = test_set.loc[np.logical_and(test_set['Region'] == r, np.logical_and(test_set['Sector'] == s, test_set['AvRating'] == ar)), 'CDS_vol'].index.values
                        proxy_series = (beta_est['Intercept'] + beta_est[r] +  
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est['Intercept vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[r + ' vol']) # all vol components
                                        
                        test_set.loc[np.logical_and(test_set['Region'] == r,
                                                    np.logical_and(test_set['Sector'] == s, 
                                                                   test_set['AvRating'] == ar)), 'CDS_vol'] = proxy_series.values  #Non vol component + vol components
                    elif r == 'Europe' and s != 'Financials' and ar != 'A':
                        test_issuers = test_set.loc[np.logical_and(test_set['Region'] == r, np.logical_and(test_set['Sector'] == s, test_set['AvRating'] == ar)), 'CDS_vol'].index.values
                        proxy_series = (beta_est['Intercept'] + beta_est[s] + beta_est[ar] +  
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est['Intercept vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[s + ' vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[ar + ' vol']) # all vol components
                                        
                        test_set.loc[np.logical_and(test_set['Region'] == r,
                                                    np.logical_and(test_set['Sector'] == s, 
                                                                   test_set['AvRating'] == ar)), 'CDS_vol'] = proxy_series.values  #Non vol component + vol components
                    elif r != 'Europe' and s == 'Financials' and ar != 'A':
                        test_issuers = test_set.loc[np.logical_and(test_set['Region'] == r, np.logical_and(test_set['Sector'] == s, test_set['AvRating'] == ar)), 'CDS_vol'].index.values
                        proxy_series = (beta_est['Intercept'] + beta_est[r] + beta_est[ar] +  
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est['Intercept vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[r + ' vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[ar + ' vol']) # all vol components
                                        
                        test_set.loc[np.logical_and(test_set['Region'] == r,
                                                    np.logical_and(test_set['Sector'] == s, 
                                                                   test_set['AvRating'] == ar)), 'CDS_vol'] = proxy_series.values  #Non vol component + vol components
                    elif r != 'Europe' and s != 'Financials' and ar == 'A':
                        test_issuers = test_set.loc[np.logical_and(test_set['Region'] == r, np.logical_and(test_set['Sector'] == s, test_set['AvRating'] == ar)), 'CDS_vol'].index.values
                        proxy_series = (beta_est['Intercept'] + beta_est[r] + beta_est[s] +  
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est['Intercept vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[r + ' vol'] +
                                        CDS_hist_temp.loc[(r,s,ar, list(test_issuers))]*beta_est[s + ' vol'] ) # all vol components
                                        
                        test_set.loc[np.logical_and(test_set['Region'] == r,
                                                    np.logical_and(test_set['Sector'] == s, 
                                                                   test_set['AvRating'] == ar)), 'CDS_vol'] = proxy_series.values  #Non vol component + vol components
        
        test_set.CDS_vol = np.exp(test_set.CDS_vol) 
        return [test_set, beta_est]
    
    def rmse(self, real_value, proxy_estimate):
        #use self.dVariable and test_set
        from sklearn.metrics import mean_squared_error
        
        return np.sqrt(mean_squared_error(real_value, proxy_estimate))
    
    def mad(self, real_value, proxy_estimate):
        'use self.dVariable and test_set'
        from sklearn.metrics import mean_absolute_error
        
        return mean_absolute_error(real_value, proxy_estimate)
    
    