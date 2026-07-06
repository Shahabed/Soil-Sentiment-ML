# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:46:49 2023

@author: Shahabedin Chatraee Azizabadi
Data preprocessing functions
"""

import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

#------------------------------------------------------------------------------------------------------------------------------

#The folloing function removies the 

# The following function can deal with the missing values like NaN, Inf,etc. DATA IMPUTATION. 
#It is for the case when we have missing values in our data.
def data_imputation(df):
    df=df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna(axis=1,how='all')
    df=pd.DataFrame(df).fillna(df.mean())
  
    return df

# Removing the mismatches from  the similarity table based on the review and report from the experts°°°°Also removing the duplicate matches which are not adding 
# any learning weights to our prediction function( Model )
def remove_duplicates(df):
    #First we remove the duplicate matches 
    df1=df.drop_duplicates( subset=[ "content_id","page_id"], keep="first", inplace=False)
    df1=df1.reset_index(drop=True)
    
    return df1

# Removing unnecessary columns
def remove_columns(df):
    list=['url','headline','text','page_id']
    df=df.drop(columns =list)
    
    return df


#--Multicolinearity measure:VIF. WE impliment the VIF to compare the level of multicolinearity. generaly gives a value for
#-- each feature. 
def VIF_computation(X):
    
    vif = pd.DataFrame()  
    vif["features"] = X.columns # Here we choose the column labels of the data frame
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif

#-----A function for the heatmap to visualise the correlation between features

def heat_map(df):
    corr = df.corr()
    sns.heatmap(corr)
    
    return
#---------a function for dimensionality reduction----------------
#-----The first candidate is using the Pearson correlation and reducing the dimension directly without changing the 
#-----content of features---------
def pearson_dim_redu(df):
    
    corr = df.corr()    
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = df.columns[columns]
    data = df[selected_columns]
    
    
    return data

