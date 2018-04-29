# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:08:50 2018

@author: Felix Farias Fueyo
"""
#Loading libraries 
import numpy as np 
import pandas as pd

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
        test_set = test_set.assign( InterEstimate = pd.Series(np.zeros(len(test_set))))
        
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
    
    def cross_section_method(self, CDS_data, test_set):
        test_set = test_set.assign( CrossEstimate = pd.Series(np.zeros(len(test_set))))
        #Log-transformation of spread
        CDS_data[self.spreadXy] = np.log(CDS_data[self.spreadXy])
        
        #Apply the same algorithm as for intersection method to create buckets and obtain estimate
        h_cat = test_set[self.index_names[0]].unique()

        for i in h_cat:
            h2_cat = []
            h2_cat = test_set.loc[test_set[self.index_names[0]] == i, self.index_names[1]].unique()
            if len(h2_cat) != 0:
                for j in h2_cat: 
                    h3_cat = []
                    h3_cat = test_set.loc[np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), self.index_names[2]].unique()
                    if len(h3_cat) != 0:
                        for m in h3_cat:
                            s_hat = CDS_data.loc[np.logical_and(np.logical_and(CDS_data[self.index_names[0]] == i, CDS_data[self.index_names[1]] == j), CDS_data[self.index_names[2]] == m), self.spreadXy]
                            test_set.loc[np.logical_and(np.logical_and(test_set[self.index_names[0]] == i, test_set[self.index_names[1]] == j), test_set[self.index_names[2]] == m), 'CrossEstimate'] = np.nanmean(s_hat)

        test_set['CrossEstimate'] = np.exp(test_set['CrossEstimate'])
        
        return test_set
    
    #Extended cross-section method with equity, volatility or equity and volatility
    def equity_method(self, CDS_data, equity_data, test_data, equity_vol_ev):
        if equity_vol_ev.size == 2:
            regressors = 'EquityVol'
        elif equity_vol_ev == 'Equity':
            regressors = 'Equity'
        elif equity_vol_ev == 'Vol':
            regressors = 'Vol'
        
        
        return test_data
    
    def rmse(self, real_value, proxy_estimate):
        #use self.dVariable and test_set
        from sklearn.metrics import mean_squared_error
        
        return np.sqrt(mean_squared_error(real_value, proxy_estimate))
    
    def mad(self, real_value, proxy_estimate):
        'use self.dVariable and test_set'
        from sklearn.metrics import mean_absolute_error
        
        return mean_absolute_error(real_value, proxy_estimate)
    
    