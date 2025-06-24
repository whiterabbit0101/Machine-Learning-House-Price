import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load csv file
df = pd.read_csv('train.csv')
print(f"Loaded dataset with {len(df)} rows and {df.shape[1]} columns")

# Check for missing values
missing_values = df.isnull().sum()

# Create a copy of the original data for cleaning
df_cleaned = df.copy()

# Use Label Encoding for categorical variables
# Identify categorical columns
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

# Apply label encoding to each categorical column
label_encoder = LabelEncoder()
for col in categorical_cols:
    # Fill missing values with 'missing' before encoding
    df_cleaned[col] = df_cleaned[col].fillna('missing')
    # Apply label encoding
    df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])

# Handle missing values in numeric columns with median imputation
numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
for col in numeric_cols:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('house_prices_cleaned.csv', index=False)

print(f"\nCleaned dataset using label encoding and median imputation. New cleaned file saved to 'house_prices_cleaned.csv'")
print(f"The cleaned dataset has {len(df_cleaned)} rows and {df_cleaned.shape[1]} columns")
print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")