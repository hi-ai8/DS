# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, classification_report

# # Create synthetic dataset
# df = pd.DataFrame({
#     'Feature1': [2, 3, 5, 7, 9, 11, 13, 15, 17, 19],
#     'Feature2': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
#     'Label': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# })

# # Split data
# X = df[['Feature1', 'Feature2']]
# y = df['Label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Logistic Regression
# log_model = LogisticRegression().fit(X_train, y_train)
# log_pred = log_model.predict(X_test)

# # Decision Tree
# tree_model = DecisionTreeClassifier().fit(X_train, y_train)
# tree_pred = tree_model.predict(X_test)

# # Evaluate both models
# print("Logistic Regression Results:")
# print(f"Accuracy: {accuracy_score(y_test, log_pred):.4f}")
# print(classification_report(y_test, log_pred, zero_division=0))

# print("\nDecision Tree Results:")
# print(f"Accuracy: {accuracy_score(y_test, tree_pred):.4f}")
# print(classification_report(y_test, tree_pred, zero_division=0))

# # Visualize Decision Tree
# plt.figure(figsize=(6, 4)) 
# plot_tree(tree_model, feature_names=['Feature1', 'Feature2'], 
#           class_names=['0', '1'], filled=True)
# plt.tight_layout()
# plt.show()


#!IRIS
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target variable (species)
df['Species'] = iris.target

# Step 2: Convert to binary classification (Class 0 vs Class 1)
# Select only class 0 and class 1 for binary classification
df_binary = df[df['Species'] < 2]

# Step 3: Define features (X) and target (y)
X = df_binary.drop(columns=['Species'])  # Features
y = df_binary['Species']  # Target (Binary outcome: Class 0 or Class 1)

# Step 4: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the Logistic Regression model
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 7: Evaluate the model using classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Output the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Step 8: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)



# #!Custom
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# # Step 1: Load your dataset
# data = {
#     'Feature1': [2, 3, 5, 7, 9, 11, 13, 15, 17, 19],
#     'Feature2': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
#     'Label':     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# }

# df = pd.DataFrame(data)

# # Step 2: Define features and target
# X = df[['Feature1', 'Feature2']]  # Independent variables
# y = df['Label']                   # Binary target variable

# # Step 3: Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Train the Logistic Regression model
# model = LogisticRegression().fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Step 6: Output
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print("\nConfusion Matrix:")
# print(conf_matrix)
