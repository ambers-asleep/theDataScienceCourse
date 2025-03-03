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

#print(data)

#print(data.head(10))
#data.info()
#data["ocean_proximity"].unique()
#missing_values = data.isnull().sum()
#missing_percentage = (missing_values / len(data)) * 100

#print("Missing values in each column: \n", missing_values)
#print("Missing data percentage: \n", missing_percentage)

data_cleaned = data.dropna()

#print("\nMissing values in each column after removal:")
#print(data_cleaned.isnull().sum())

#print(data.describe())

sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
sns.histplot(data_cleaned['median_house_value'], color = 'forestgreen', kde=True)
plt.title("Distribution of Median House Values")
plt.xlabel("Median House Values")
plt.ylabel("Frequency")
plt.show()


#end