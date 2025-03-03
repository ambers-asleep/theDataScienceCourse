#Begin with problem identification 
#define variables
#import libraries 
#process data 
#after purifying your data model

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#import files and define path
#place data in pd.read_csv(file_path)
data = pd.read_csv("machineLearningRealEstateProject\housing.csv")
data_cleaned = data.dropna()

#assign some dummys
ocean_proximity_dummies = pd.get_dummies(data['ocean_proximity'], prefix='ocean_proximity')
data = pd.concat([data.drop("ocean_proximity", axis = 1), ocean_proximity_dummies], axis = 1)
print(ocean_proximity_dummies)
data = data.drop("ocean_proximity_ISLAND", axis = 1)
data.columns

#Train the model


#end
