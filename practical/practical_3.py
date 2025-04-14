import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 32, 47, 51],
    'Salary': [50000, 54000, 82000, 90000],
    'Department': ['HR', 'IT', 'Finance', 'IT']
}

df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Feature Scaling
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

df['Age_normalized'] = minmax_scaler.fit_transform(df[['Age']])
df['Salary_standardized'] = standard_scaler.fit_transform(df[['Salary']])

# Feature Dummification
df_dummies = pd.get_dummies(df['Department'], prefix='Dept')
df = pd.concat([df, df_dummies], axis=1)

print("\nDataFrame after Scaling and Dummification:\n", df)
