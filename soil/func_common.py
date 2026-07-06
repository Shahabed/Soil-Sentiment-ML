#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Shahabedin Chatraee Azizabadi
prediction model:common functions
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
#_____________ Data preprocessing before model creation______________________

#scaling of the design matrix
def scal_d_matrix(df):
    scaler = MinMaxScaler()
    scal_df=scaler.fit_transform(df)
    
    return scal_df


#Label vector generation
def generate_y_vector(df):
    vectors = df['soc_percent_labval']
    return vectors



# test_train spliting
def spliting_for_classifier(features_matrix,labels):
    
    (X, y) = (features_matrix, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    
    return X_train, X_test, y_train, y_test    

# Checking for multicollinearity: Variance Inflation Factor
def VIF_computation(X):
    
    vif = pd.DataFrame()  
    vif["features"] = X.columns # Here we choose the column labels of the data frame
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif

# Principal component analysis: solving the problem of multicollinearity, also reducing domension of features space
def PCA_al(X):
    pca = PCA(n_components=170)
    X_principalComponents = pca.fit_transform(X)
    return X_principalComponents

#------------Evaluation metrics: function to evaluate regression algorith performance--------------------------
# A function to X_test and y_test for each of the separate fields 
def field_X_test_and_y_test(df):
    X = df.loc[:, 'feature1':]
    y =df.soc_percent_labval    
        
    return X, y 
   
# Some metrics for the eavluation of regression results
def evaluate_model(y_test, y_predict):
    evs = metrics.explained_variance_score(y_test, y_predict)
    r2_score = metrics.r2_score(y_test, y_predict)    
    mse = metrics.mean_squared_error(y_test, y_predict)
    mae = metrics.mean_absolute_error(y_test, y_predict)
    
    print('Mean Absolute Error:', round(mae, 2), 'degrees.')
    print('Mean Square Error:', round(mse, 2))
    print("score (r2_score):", r2_score)
    print("explained variance score:", evs)    
