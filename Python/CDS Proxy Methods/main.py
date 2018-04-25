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
CDS_data = pd.read_csv('Data/CDS_data.csv')

### Pre-Process ###


### Cross-Section Testing ###

#1. Create N data sets.

#2. Apply the methodologies to every possible N-1 data set.

#3. Test their accuracy on the 'left-one-out' subsets.