#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Shahabedin Chatraee Azizabadi
prediction model of soil organic carbon(SOC)
"""

import numpy as np
import pandas as pd
import func_common
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer


# Reading the result of the initial analysis-----------------------------------
df=pd.read_csv("output_analysis_d.csv")

design_matrix = df.loc[:, 'nir_1374':]

# Scaling the design matrix using MinMaxScaler-----------------------------------
df_scaled = func_common.scal_d_matrix(design_matrix)

# Check for the variance inflation factor: detecting Multicollinearity---------
# v = func_common.VIF_computation(design_matrix)
# colum_small_vif = v.loc[(v['VIF Factor'] < 10)]# ALL FEATURES HAVE THE Multicollinearity
# v.to_csv("Variance_Inflation_Factor.csv", index=False)

# Principal component analysis: solving the problem of Multicollinearity--------
X = func_common.PCA_al(df_scaled)

# Creating a new dataframe from the new features: 
# we will use it later: for the regression-percision test on different fields
col_list = ['feature' + str(x) for x in range(1,171)]
df_new = pd.concat([df.iloc[:, 0:5], pd.DataFrame(X, columns=col_list)], axis=1)

# Creating the target vector --------------------------------------------------
y = func_common.generate_y_vector(df)

# Train-test spliting
X_train, X_test, y_train, y_test = func_common.spliting_for_classifier(X, y)#df_scaled, y)


#--------------- Defining and training the prediction model--------------------
reg = LinearRegression().fit(X_train, y_train)

# For comparision, support vector machine regression algorithm was tested
#reg = svm.SVR(gamma='auto').fit(X_train, y_train)

#------------- Boosting algorithm: did not change the performance of the regression model
# model = LinearRegression() #RandomForestRegressor()
# reg = TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)

#---------Prediction for the test data set--------------------------------------
y_pred = reg.predict(X_test)

# ----------------- Metrics for evaluating our trained regression model -------------
#   Explained variance score: evs= 0.71 acceptable result for the regression model
print("Evaluattion of the regression model using the data of all fields together")
func_common.evaluate_model(y_test, y_pred)

# Ploting the result of prediction for entire data
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Expected', lw=4)
ax.set_xlabel('real data')
ax.set_ylabel('Predicted data')
ax.legend(loc='best')
plt.show()

#------ To check the generalization of the prediction model, we test the trained model on three separate fields--------------
print("Now tetsing the trained model on the data of each field")
for field_label in ['field_A', 'field_B', 'field_C']:
    field_df = df_new[df_new.label_location == field_label]
    print()
    print("--"+field_label+"--")
    Xt, yt = func_common.field_X_test_and_y_test(field_df)
    yt_predict = reg.predict(Xt)
    func_common.evaluate_model(yt, yt_predict)
