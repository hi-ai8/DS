import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
df_csv = pd.read_csv('data.csv')
print("CSV Data:\n", df_csv.head())

# Read JSON file
df_json = pd.read_json('data.json')
print("\nJSON Data:\n", df_json.head())

# ===== Basic Preprocessing on CSV =====

# 1. Handling Missing Values
print("\nMissing Values:\n", df_csv.isnull().sum())

# Fill missing salary with mean
if 'salary' in df_csv.columns:
    df_csv['salary'] = df_csv['salary'].fillna(df_csv['salary'].mean())

# Drop rows with missing values (if any)
df_csv = df_csv.dropna()
print("\nData after handling missing values:\n", df_csv)

# 2. Handling Outliers (IQR method)
if 'salary' in df_csv.columns:
    Q1 = df_csv['salary'].quantile(0.25)
    Q3 = df_csv['salary'].quantile(0.75)
    IQR = Q3 - Q1

    df_csv = df_csv[~((df_csv['salary'] < (Q1 - 1.5 * IQR)) | (df_csv['salary'] > (Q3 + 1.5 * IQR)))]
    print("\nData after removing outliers:\n", df_csv)

# ===== Data Transformation =====

# 3. Filtering rows where age > 25
if 'age' in df_csv.columns:
    filtered_df = df_csv[df_csv['age'] > 25]
    print("\nFiltered Data (age > 25):\n", filtered_df)

# 4. Sorting by age
if 'age' in df_csv.columns:
    sorted_df = df_csv.sort_values(by='age')
    print("\nSorted Data by Age:\n", sorted_df)

# 5. Grouping by gender and averaging salary
if 'gender' in df_csv.columns and 'salary' in df_csv.columns:
    grouped = df_csv.groupby('gender')['salary'].mean()
    print("\nGrouped Data (Average Salary by Gender):\n", grouped)

# ===== Visualization (Optional) =====

# Visualize missing values (from original data before preprocessing)
sns.heatmap(pd.read_csv('data.csv').isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
