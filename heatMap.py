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
data["ocean_proximity"].unique()
data_cleaned = data.dropna()
data_cleaned = data_cleaned.drop('ocean_proximity', axis=1)

#remove outliers
Q1 = data_cleaned['median_house_value'].quantile(0.25)
Q3 = data_cleaned['median_house_value'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_no_outliers_1 = data_cleaned[(data_cleaned['median_house_value'] >= lower_bound) & (data_cleaned['median_house_value'] <= upper_bound)]

#print("Original data shape:", data_cleaned.shape)
#print("New data shape without outliers:", data_no_outliers_1.shape)

#begin boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x=data_no_outliers_1['median_income'], color='purple')
plt.title('Outlier Analysis in Median Income')
plt.xlabel("Median Income")
plt.show()

#calculate Q1 and Q3 and IQR
Q1 = data_cleaned['median_income'].quantile(0.25)
Q3 = data_cleaned['median_income'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_no_outliers_2 = data_no_outliers_1[(data_no_outliers_1['median_income'] >= lower_bound) & (data_no_outliers_1['median_income'] <= upper_bound)]
data_cleaned = data_no_outliers_2

print("Original data shape: ", data_no_outliers_1.shape)
print("New shape without outliers: ", data_no_outliers_2.shape)

plt.figure(figsize=(10,6))
sns.boxplot(x=data_no_outliers_2['median_income'], color='purple')
plt.title('Outlier Analysis in Median Income')
plt.xlabel("Median Income")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(data_cleaned.corr(), annot=True, cmap='Greens')
plt.title('Correlation Heatmap of Housing Data')
plt.show()

for column in ['ocean_proximity']:
    print(f"Unique values in {column}: ", data[column].unique())

#end
