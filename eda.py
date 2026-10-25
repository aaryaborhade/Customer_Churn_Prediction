import pandas as pd

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Churn distribution
print("\nChurn Distribution:")
print(df["Exited"].value_counts())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())