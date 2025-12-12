# Step 1: Import libraries and setup configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot') # Use a style named ggplot, a style mainly used in R for better styling
pd.set_option('display.max_columns', 100) # Limit the column size

# Step 2: Load dataset and classify them based on their types
df = pd.read_csv(r"cleveland.csv", na_values="?") # Load csv, since the dataset treats na values as ?, we'll adjust to it
print(df.head()) # Read first 5 records from the dataset
print(np.shape(df))
print(df.value_counts())
print(df.describe()) # Get the details statistics of each column in dataset like standard deviation, mean, minimum and maximum value
print(f"\nNumerical features in dataset: {df.select_dtypes(exclude='object').columns.tolist()}")
print(f"\nCategorical features in dataset: {df.select_dtypes(include='object').columns.tolist()}")

# Step 3: Clean dataset by dropping NaN values
print(df.isna().sum()) # Calculate the number of records in each column that contains NaN value
df.dropna(inplace=True) # Remove all records with NaN value, the difference between inplace=True or without it is that inplace=True directly remove NaN value from dataset, whereas dropna() alone will create a copy and return the copy dataframe that are dropped. Thus with dropna() you need to assign the original df to the new returned df to save the record
print(np.shape(df))

# Step 4: Feature understanding and visualisation
# 4.1a Box plot with outliers for resting blood pressure
sns.boxplot(data=df, x='trestbps').set_title("Box plot for resting blood pressure with outliers")
plt.show()

# 4.1a Box plot without outliers for resting blood pressure
resting_blood_pressure_outliers = df['trestbps'].quantile(0.95) # Get the value in the column for the 95th percentile
print(resting_blood_pressure_outliers) # Display output (160.8)
filtered_df = df[
    df['trestbps'] < resting_blood_pressure_outliers
]
sns.boxplot(data=filtered_df, x='trestbps').set_title("Box plot for resting blood pressure without outliers") # Remove all resting blood pressure data above the 95th percentile which is 160.8
plt.show()

# 4.2 Histogram for Cholesterol level
# 4.2a Cholesterol Histogram with outliers
sns.histplot(data=df, x='chol').set_title("Histogram for cholesterol level with outliers")
plt.show()
# 4.2b Cholesterol Histogram without outliers
cholesterol_outliers = df['chol'].quantile(0.95)
cholesterol_outliers2 = df['chol'].quantile(0.05)
print(cholesterol_outliers)
filtered_df = df[
    (df['chol'] > cholesterol_outliers2) &
    (df['chol'] < cholesterol_outliers)
]
sns.histplot(data=filtered_df, x='chol').set_title("Histogram for cholesterol level without outliers")
plt.show()

# 4.3 Histogram for maximum heart rate value
# 4.3a Maximum Heart Rate Histogram with outliers
sns.histplot(data=df, x='thalach').set_title("Histogram for maximum heart rate with outliers")
plt.show()
# 4.3b Maximum Heart Ratte Histogram without outliers
maximum_heart_rate_outliers = df['thalach'].quantile(0.05)
maximum_heart_rate_outliers2 = df['thalach'].quantile(0.95)
print(maximum_heart_rate_outliers)
print(maximum_heart_rate_outliers2)
filtered_df = df[
    (df['thalach'] > maximum_heart_rate_outliers) &
    (df['thalach'] < maximum_heart_rate_outliers2)
]
sns.histplot(data=filtered_df, x='thalach').set_title("Histogram for maximum heart rate without outliers")
plt.show()

# Scatterplot between age and heart disease label
sns.scatterplot(data=df, x="age", y="trestbps", hue="num", palette="tab10").set_title("Relationship between age and heart disease results")
plt.show()

# Heatmap
df_corr = df.corr()
sns.heatmap(data=df_corr, annot=False, cmap="YlGnBu").set_title("Correlation Matrix Heatmap")
plt.show()

