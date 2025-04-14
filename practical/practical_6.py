import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Dataset (you can replace this with CSV/Excel file reading)
data = {
    'Hours_Studied': [2, 4, 6, 8, 10, 12],
    'Attendance': [60, 65, 70, 75, 80, 85],
    'Marks': [50, 55, 65, 70, 75, 85]
}

df = pd.DataFrame(data)

# --------------------
# Simple Linear Regression (Hours_Studied → Marks)
# --------------------
X_simple = df[['Hours_Studied']]
y = df['Marks']

model_simple = LinearRegression()
model_simple.fit(X_simple, y)

print("🔹 Simple Linear Regression:")
print(f"Intercept: {model_simple.intercept_}")
print(f"Coefficient: {model_simple.coef_[0]}")
y_pred_simple = model_simple.predict(X_simple)
print(f"R² Score: {r2_score(y, y_pred_simple)}")

# Plotting the Simple Regression Line
plt.scatter(df['Hours_Studied'], y, color='blue', label='Actual Marks')
plt.plot(df['Hours_Studied'], y_pred_simple, color='red', label='Predicted Line')
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# --------------------
# Multiple Linear Regression (Hours_Studied + Attendance → Marks)
# --------------------
X_multi = df[['Hours_Studied', 'Attendance']]
model_multi = LinearRegression()
model_multi.fit(X_multi, y)

print("\n🔹 Multiple Linear Regression:")
print(f"Intercept: {model_multi.intercept_}")
print(f"Coefficients: {model_multi.coef_}")  # [coef1, coef2]
y_pred_multi = model_multi.predict(X_multi)
print(f"R² Score: {r2_score(y, y_pred_multi)}")

# Mean Squared Error
mse = mean_squared_error(y, y_pred_multi)
print(f"Mean Squared Error: {mse}")
