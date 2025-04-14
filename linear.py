import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create synthetic data
data = pd.DataFrame({
    'X': np.arange(1, 11),
    'Y': [2.3, 2.5, 3.1, 3.8, 4.2, 5.1, 5.8, 6.5, 7.1, 7.8],
    'Z': [1, 3, 2, 5, 4, 6, 8, 7, 9, 10]
})

# Simple Linear Regression
X_simple = data[['X']]
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print simple regression results
print("Simple Regression:")
print(f"Equation: Y = {model.intercept_:.2f} + {model.coef_[0]:.2f}X")
print(f"R²: {r2_score(y_test, y_pred):.4f}, MSE: {mean_squared_error(y_test, y_pred):.4f}")

# Plot regression line
plt.figure(figsize=(6, 4))
plt.scatter(data['X'], data['Y'])
plt.plot(data['X'], model.predict(data[['X']]), color='red')
plt.title('Simple Linear Regression')
plt.tight_layout()
plt.show()

# Multiple Linear Regression
X_multi = data[['X', 'Z']]
X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression().fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

# Print multiple regression results
print("\nMultiple Regression:")
print(f"Equation: Y = {model_multi.intercept_:.2f} + {model_multi.coef_[0]:.2f}X + {model_multi.coef_[1]:.2f}Z")
print(f"R²: {r2_score(y_test, y_pred_multi):.4f}, MSE: {mean_squared_error(y_test, y_pred_multi):.4f}")

#!Multiple Regression
# import pandas as pd 
# from sklearn.linear_model import LinearRegression 
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import mean_squared_error, r2_score 
 
# # Step 1: Create dataset 
# data = { 
#     'Bedroom': [3, 3, 2, 4, 3, 4, 3, 3, 3, 3, 3], 
#     'Bathroom': [1, 2.25, 1, 3, 2, 4.5, 2.25, 1.5, 1, 2.5, 2.5], 
#     'Sqft_living': [1180, 2570, 770, 1960, 1680, 5420, 1715, 1060, 1780, 1890, 
# 3560], 
#     'Floors': [1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1], 
#     'Grade': [7, 7, 6, 7, 8, 11, 7, 7, 7, 7, 8], 
#     'Sqft_above': [1180, 2170, 770, 1050, 1680, 3890, 1715, 1060, 1050, 1890, 1860], 
#     'Sqft_basement': [0, 400, 0, 910, 0, 1530, 0, 0, 730, 0, 1700], 
#     'Price': [221900, 538000, 180000, 604000, 510000, 267800, 257500, 291850, 
# 229500, 323000, 662500] 
# } 
 
# df = pd.DataFrame(data) 
 
# # Step 2: Define features and target 
# X = df.drop(columns='Price')  # All columns except Price 
# y = df['Price']               # Target is Price 
 
# # Step 3: Split into train and test 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
# random_state=42) 
 
# # Step 4: Train the model 
# model = LinearRegression().fit(X_train, y_train)
# y_pred = model.predict(X_test)
 
# print("\nMultiple Regression:")
# print(f"R²: {r2_score(y_test, y_pred):.4f}, MSE: {mean_squared_error(y_test, y_pred):.4f}")

# #!IRIS
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris

# # Load the Iris dataset
# iris = load_iris()
# df = pd.DataFrame(iris.data, columns=iris.feature_names)

# # Add target variable to the DataFrame (species)
# df['species'] = iris.target

# # Independent variable (petal.length)
# X = df[['petal length (cm)']]  # Independent variable

# # Dependent variable (petal.width)
# y = df['petal width (cm)']  # Dependent variable

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the linear regression model
# model = LinearRegression().fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Output results
# print("\nMultiple Regression:")
# print(f"R²: {r2_score(y_test, y_pred):.4f}, MSE: {mean_squared_error(y_test, y_pred):.4f}")

# # Plot the regression line
# plt.scatter(X_test, y_test, color='blue', label='Actual values')
# plt.plot(X_test, y_pred, color='red', label='Regression line')
# plt.xlabel('Petal Length (cm)')
# plt.ylabel('Petal Width (cm)')
# plt.title('Linear Regression: Petal Length vs Petal Width')
# plt.legend()
# plt.show()


# #!salary
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # Dataset: Years of Experience and Salary
# data = {
#     'Years of Experience': [2, 10, 4, 20, 8, 12, 22],
#     'Salary': [30000, 95000, 45000, 178000, 84000, 120000, 200000]
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Define the independent variable (X) and dependent variable (y)
# X = df[['Years of Experience']]  # Independent variable
# y = df['Salary']  # Dependent variable

# # Split the data into training and testing sets (80% training, 20% testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a linear regression model
# model = LinearRegression().fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Output results
# print("\nMultiple Regression:")
# print(f"R²: {r2_score(y_test, y_pred):.4f}, MSE: {mean_squared_error(y_test, y_pred):.4f}")

# # Plotting the regression line
# plt.scatter(X_test, y_test, color='blue', label='Actual values')
# plt.plot(X_test, y_pred, color='red', label='Regression line')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.title('Linear Regression: Years of Experience vs Salary')
# plt.legend()
# plt.show()

