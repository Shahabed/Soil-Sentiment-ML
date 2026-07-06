#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Shahabedin Chatraee Azizabadi
prediction model of the soil organic carbon(SOC) content: initial analysis
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


#------------------Reading the dataframe-----------------------------
df = pd.read_csv("Soil_data.csv", sep=";", index_col=False)
# Original data: 1588 x 479
df = df.drop(columns=['Unnamed: 0'])


#------------------------Uniqueness of measurments-----------------------------
# Checking the uniqness of measurment_id: **They are unique**
df['measurement_ID'].nunique()

#------------------------------Data cleaning-----------------------------------
# See the rows who have any Nan or Inf cell
T = df[df.isin([np.nan, np.inf, -np.inf]).any(1)]
# Only 10 rows, so we decide to just drop them
df.replace([np.inf, -np.inf], np.nan)
print(df)
df.dropna(inplace=True)


# Check for the  Null values
df[df.isnull().any(1)]
# nothing left

#Filtering records with lat,long of ZERO
df = df[df.lat != 0]
# measurement_ID	lat	lng	label_location
# 	bknrffdbk	0.0	0.0	field_A
# 	kfoffsosf	0.0	0.0	field_B
# 	rbsofffob	0.0	0.0	field_C

#----------------More investigation on data-----------------------------------
df.info()
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 1575 entries, 0 to 1587
# Columns: 478 entries, measurement_ID to uv_829
# dtypes: float64(475), object(3)
# memory usage: 5.8+ MB

# Let's find out which are those three 'object' columns
df.dtypes[df.dtypes == 'object']
# measurement_ID        object
# label_location        object
# soc_percent_labval    object

#****** PROBLEM DETECTED!******
# soc_percent_labval should be numeric not object
# it has one incorrect entry "<0.6"
# We ewmove that row
df = df[df.soc_percent_labval != '<0.6']
# Now, change its type to numerical
df.soc_percent_labval = df.soc_percent_labval.astype(float)

#------------further analysis of target variable--> SOC------------------------
# Comparing to the normal distribution
stats.probplot(df.soc_percent_labval, dist='norm', plot=plt)
plt.show()
#For disnity function of SOC
# a=df.soc_percent_labval.to_frame()
# a.plot.density()
# plt.show()
# By looking at SOC distribution, we see an outlier biger than 9: remove the outlier
# Also the distribution of SOC is close to a normal distribution
df = df[df.soc_percent_labval < 9]
df.to_csv("output_analysis_d.csv", index=False)
# --------------------------------------------------
#--------------------Data visualization----------------------------------------
# Visualizing the spatial location of data points in three fields
field_A = df[df.label_location == 'field_A']
field_B = df[df.label_location == 'field_B']
field_C = df[df.label_location == 'field_C']

plt.scatter(field_A.lng, field_A.lat, color='red', marker='.', label='field_A')
plt.scatter(field_B.lng, field_B.lat, color='green', marker='.',label='field_B')
plt.scatter(field_C.lng, field_C.lat, color='blue', marker='.', label='field_C')
plt.xlabel('Longitude',fontsize=14)
plt.ylabel('Latitude',fontsize=14)
plt.legend(loc='best')
plt.show()

# Demonastration of SOC level for three fields
plt.scatter(df.label_location, df.soc_percent_labval)
plt.xlabel('Different_fields',fontsize=14)
plt.ylabel('SOC_content',fontsize=14)
plt.show()
