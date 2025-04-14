import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset and create sample categorical column
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['house_category'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df))

# Feature Scaling
numeric_columns = df.select_dtypes(include=np.number).columns

# 1. Standardization (Z-score)
std_scaler = StandardScaler()
df_std = df.copy()
df_std[numeric_columns] = std_scaler.fit_transform(df[numeric_columns])

# 2. Min-Max Normalization 
mm_scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[numeric_columns] = mm_scaler.fit_transform(df[numeric_columns])

# 3. One-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=['house_category'], drop_first=True)

# Show results (first 3 rows only)
print("Standardized:\n", df_std.head(3))
print("\nNormalized:\n", df_norm.head(3))
print("\nOne-Hot Encoded:\n", df_encoded.head(3))