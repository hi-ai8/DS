import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df_csv = pd.read_csv('data.csv')
df_json = pd.read_json('data.json')

# Quick Data Exploration
print(df_csv.info())
print(df_csv.describe())

# Handle Missing Values - Simplified approach
df_csv.fillna(df_csv.mean(numeric_only=True), inplace=True)
df_csv.ffill(inplace=True)  # Forward fill remaining NAs

# Handle Outliers - Using IQR method for numeric columns
for col in df_csv.select_dtypes(include=np.number).columns:
    q1, q3 = df_csv[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_csv = df_csv[(df_csv[col] >= lower_bound) & (df_csv[col] <= upper_bound)]

# Basic Data Manipulation - Only perform if columns exist
if 'salary' in df_csv.columns and 'department' in df_csv.columns:
    # Filter, sort and group operations
    filtered_df = df_csv[df_csv['salary'] > 10]
    sorted_df = df_csv.sort_values(by='salary')
    grouped_df = df_csv.groupby('department').mean(numeric_only=True)

# Save Cleaned Data
df_csv.to_csv('cleaned_data.csv', index=False)
df_json.to_json('cleaned_data.json', orient='records')

# Simple Visualization
if 'salary' in df_csv.columns:
    plt.figure(figsize=(10, 4))
    df_csv['salary'].hist(bins=20)
    plt.title('Salary Distribution')
    plt.show()

    # Correlation heatmap of numeric columns
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_csv.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.show()


# #!data    
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load Data
# df_csv = pd.read_csv('Country_Age_Salary_Purchased.csv')

# # Quick Data Exploration
# print("Dataset Information:")
# print(df_csv.info())
# print("\nDescriptive Statistics:")
# print(df_csv.describe())

# # Handle Missing Values
# # For numeric columns (Age, Salary) - use mean imputation
# df_csv['Age'].fillna(df_csv['Age'].mean(), inplace=True)
# df_csv['Salary'].fillna(df_csv['Salary'].mean(), inplace=True)
# # For categorical columns - use forward fill
# df_csv.ffill(inplace=True)

# # Handle Outliers - Using IQR method for numeric columns
# numeric_cols = ['Age', 'Salary']
# for col in numeric_cols:
#     q1, q3 = df_csv[col].quantile([0.25, 0.75])
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
    
#     # Identify outliers
#     outliers = df_csv[(df_csv[col] < lower_bound) | (df_csv[col] > upper_bound)]
#     if not outliers.empty:
#         print(f"\nOutliers in {col}:")
#         print(outliers)
    
#     # Filter out outliers
#     df_csv = df_csv[(df_csv[col] >= lower_bound) & (df_csv[col] <= upper_bound)]

# print("\nDataset shape after removing outliers:", df_csv.shape)
# df_csv.to_csv('cleaned_data_country.csv', index=False)