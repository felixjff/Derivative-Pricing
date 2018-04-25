# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:08:50 2018

@author: Felix Farias Fueyo
"""
#Loading libraries 
import numpy as np 
import pandas as pd

class ProxyMethods:
    
    def __init__(self, num_index, index_names, spreadXy):
        #number of indeces user to divide data set/create buckets (numeric)
        self.num_index = num_index
        #name of indeces used to create buckets (vector)
        self.index_name
        #spread maturity, where spreadXy = the X year spread. (string 'spread{int}y')
        self.spreadXy = spreadXy 
        
    #Prepare the data set for proxy modeling
    def pre_process(self, CDS_data, low_rating, currency, simple_regions, no_gov, simple_cat, bps_cap):
        CDS_data = pd.DataFrame(CDS_data)
        
        #Remove CCC and D rated credits. Their spreads considered outliers.
        if low_rating == True:
           CDS_data = CDS_data.loc[np.logical_and(CDS_data['AvRating'] != 'CCC', CDS_data['AvRating'] != 'D')]
        
        #Retain quotes in currencies according to geographic regions
        if currency == True:
            eu_countries = ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belgium','Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy','Lithuania','Luxembourg','The Netherlands','Norway','Poland','Portugal','Spain','Sweden','Switzerland','United Kingdom']
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
            cyclical = ['Basic Materials', 'Consumer Services', 'Energy', 'Technology', 'Telecommunication services', 'Industrials']
            for i in cyclical:
                CDS_data = CDS_data.loc[CDS_data['Sector'] == i] = 'Cyclical'
            
            non_cyclical = ['Consumer Goods', 'Healthcare', 'Utilities']
            for i in non_cyclical:
                CDS_data = CDS_data.loc[CDS_data['Sector'] == i] = 'Non Cyclical'
        
        #Cap the max. spread to 1000 bps.
        if bps_cap == True:
            CDS_data = CDS_data.loc[CDS_data[self.spreadXy] < 0.10]
            
        return CDS_data
        
    #normalize using simplified approach. Extension possible for hazard rate approach
    def noralize(self, RR, CDS_spread):
        norm_RR = CDS_spread*0.6/(1-RR)
        
        return norm_RR
    
    #Intersection method
    def intersection_model(self, CDS_data):
        #Create buckets, e.g. with region, sector and rating as indeces.
        i_CDS_data = pd.DataFrame(CDS_data).groupby(by = self.index_name)
        
        
        
        
    
    