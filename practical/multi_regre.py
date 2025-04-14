import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Step 1: Create the dataset manually from the image
data = {
    'Bedrooms': [3, 3, 2, 4, 3, 4, 3, 3, 3, 3],
    'Bathrooms': [1, 2.25, 1, 3, 2, 4.5, 2.25, 1.5, 1, 2.5],
    'Sqft_living': [1180, 2570, 770, 1960, 1680, 5420, 1715, 1060, 1780, 1890],
    'Floors': [1, 2, 1, 1, 1, 1, 2, 1, 1, 2],
    'Grade': [7, 7, 6, 7, 8, 11, 7, 7, 7, 8],
    'Sqft_above': [1180, 2170, 770, 1050, 1680, 3890, 1715, 1060, 1050, 1890],
    'Sqft_basement': [0, 400, 0, 910, 0, 1530, 0, 0, 730, 0],
    'Price': [221900, 538000, 180000, 604000, 510000, 267800, 257500, 291850, 229500, 323000]
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)

# Step 3: Define features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Step 4: Train the model
model = LinearRegression()
model.fit(X, y)

# Step 5: Make predictions
y_pred = model.predict(X)

# Step 6: Evaluate the model --(optional)
print("Model Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef:.2f}")
print("\n🔹 Multiple Linear Regression:")
print(f"\nIntercept: {model.intercept_:.2f}")
print(f"Coefficients: {model.coef_}") 
print(f"\nR² Score: {r2_score(y, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred):.2f}")
