# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:04:39 2023

@author: Shahabedin Chatraee Azizabadi

functions for different classification algorithms
"""

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm,naive_bayes,linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#-----------------------------------------------------------------------------------------------------------
#---------Converting Categorical Data Columns to Numerical-------------
def encoding_categ_data(df):
    columns = ['department','premium','category','title_type',
                            'title_emotion','title_clickbait',
                            'title_sentiment','subject','place','finished',
                            'liveticker','ressort','main_category','subcategory']
    
    encoder = LabelEncoder()
    for column in columns:
        df[column] = encoder.fit_transform(df[column])
    df1=df
    return df1
#--------------<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>-------------
#A function for the spliting of the design matrix and the label vector
# def spliting_for_classifier(features_matrix,labels):
    
#     (X, y) = (features_matrix, labels)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
#     return X_train, X_test, y_train, y_test
#--------------------------------------------------------
#--------maximum entropy classifier----------------
def maxEnt_classifier(X_train, X_test, y_train, y_test):
    # Making  an instance of the classification Model
    maxent = linear_model.LogisticRegression(penalty='l2', C=1.0,max_iter=100,dual=False)
    maxent.fit(X_train, y_train)
    #That is a matrix with the shape (n_classes, n_features):Coefficient of the features in the decision function can be obtain as follow:
    cof=maxent.coef_
    #Predict the class labels for samples in X
    y_predicted = maxent.predict(X_test)
    #>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    con_mat=metrics.confusion_matrix(y_test,y_predicted)
    # Here we can compute the f1 function
    f1= metrics.f1_score(y_test, y_predicted, average=None) 
    Accuracy=metrics.accuracy_score(y_test, y_predicted)
    return con_mat,cof,Accuracy,f1

#-------------random forest classifier-----------------
def random_forest_classifier(X_train, X_test, y_train, y_test):
    
    Randomf = RandomForestClassifier(n_estimators=20,criterion = 'entropy', random_state=42)
    Randomf.fit(X_train, y_train)
    y_pred = Randomf.predict(X_test)    
    con_mat=metrics.confusion_matrix(y_test,y_pred)
    class_report=metrics.classification_report(y_test,y_pred)
    Accuracy=metrics.accuracy_score(y_test, y_pred)
    return con_mat,class_report,Accuracy

#--------------------support vector machine classifier-------------------
def SVM_classifier(X_train, X_test, y_train, y_test):
    svclassifier = svm.SVC(kernel='poly', degree=3, C=1,decision_function_shape='ovo')
    svclassifier.fit(X_train, y_train)
    y_pred2 = svclassifier.predict(X_test)
    con_mat2=metrics.confusion_matrix(y_test,y_pred2)
    Accuracy2=metrics.accuracy_score(y_test, y_pred2)
    return con_mat2,Accuracy2
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<------------------
# A function for ploting confusion matrix based on the different classifiers(clf) which is given as input
def plot_confusion_matrix(clf, X_test, y_test):
    
    ax=metrics.plot_confusion_matrix(clf, X_test, y_test)
    
    return ax