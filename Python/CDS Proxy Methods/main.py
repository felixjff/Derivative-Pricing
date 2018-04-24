# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:01:52 2018

@author: Felix Farias Fueyo
"""

#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime

plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

#Loading Data
dataset = pd.read_csv('Data/CDS_data.csv')

#Normalizing spreads
