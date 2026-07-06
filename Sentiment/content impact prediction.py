# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:00:11 2023

@author: sazizaba
content impact(sentiment) prediction
"""
# reading data
import pandas as pd
import numpy as np
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import data_prep_functions
import classification_functions






#-----------------1.START OF COMPUTATION----------------------------
time_started = time.time()
#---------Read the dataframe----------------------------
df=pd.read_csv('BILD articles and sentiment monthly.csv')

#df['content_id'].nunique()
#nique_id = df['content_id','page-id'].unique()



#--------------2.data preprocessing--------------------------------------------
#---remove duplicates based on content_id and page_id
df1=data_prep_functions.remove_duplicates(df)
#
#Removing the unnecessary columns
df2=data_prep_functions.remove_columns(df1)
#
#encoding the categorical data
df3=classification_functions.encoding_categ_data(df2)
#

#------------------3.Dimensionality reduction----------------------------------
#Investigating the correlation between features by heatmap
h=data_prep_functions.heat_map(df3)
#
# Using the Pearson correlation function for dimensionality redoction with ''Correlation Threshold=0.9''--->90%
df4=data_prep_functions.pearson_dim_redu(df3)
#
# Checking for multicoliniarity with VIF factor



#------------------4.Preparing data for classification-------------------------
# creating design matrix and label vector
design_matrix=df4.loc['']
labels=df4['score']
labelencoder = LabelEncoder()
labels=labelencoder.fit_transform(labels)
sc = StandardScaler()
scaler = MinMaxScaler()
features_matrix_non_negative=scaler.fit_transform(design_matrix)
(X, y) = (features_matrix_non_negative, labels)
#
#Test and training spliting: One can use built in function or the defined function in ''classification_functions.py''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#----                                                -----------
#---------                                        --------------
#--------- 5.Training of the classification algorithms ---------------
# Testing maxEnt
#cof,dev_acc,Accuracy,f1=classification_functions.maxEnt_classifier(X_train, X_test, y_train, y_test)

#
#Testing Random forest
con_mat,class_report,Accuracy=classification_functions.random_forest_classifier(X_train, X_test, y_train, y_test)

#
# Testing SVM
#con_mat,Accuracy,f1=classification_functions.SVM_classifier(X_train, X_test, y_train, y_test)


#-----------------------------
#-----------------END OF COMPUTATION-------------------------
time_end = time.time()
print("Elapsed", np.round(time_end-time_started, 2), "seconds")