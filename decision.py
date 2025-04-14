
# #!custom
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, export_text
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Custom data
# data = {
#     'Feature1': [2, 3, 5, 7, 9, 11, 13, 15, 17, 19],
#     'Feature2': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
#     'Label':    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# }

# # Create DataFrame
# df = pd.DataFrame(data)
# X = df[['Feature1', 'Feature2']]
# y = df['Label']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluation
# print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
# print(f"Precision: {precision_score(y_test, y_pred):.2f}")
# print(f"Recall   : {recall_score(y_test, y_pred):.2f}")
# print(f"F1 Score : {f1_score(y_test, y_pred):.2f}")

# # Show decision rules
# print("\nDecision Tree Rules:")
# print(export_text(model, feature_names=X.columns.tolist()))


# #!TREE
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, export_text
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.datasets import load_iris

# # Load the Iris dataset
# iris = load_iris()
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['species'] = iris.target

# # Features and target
# X = df[iris.feature_names]
# y = df['species']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = DecisionTreeClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluation
# print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
# print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
# print(f"Recall   : {recall_score(y_test, y_pred, average='weighted'):.2f}")
# print(f"F1 Score : {f1_score(y_test, y_pred, average='weighted'):.2f}")

# # Show decision rules
# print("\nDecision Tree Rules:")
# print(export_text(model, feature_names=iris.feature_names))


#!Titantic
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Load and preprocess Titanic dataset
df = sns.load_dataset('titanic')[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Features and target
X = df[['pclass', 'sex', 'age', 'fare']]
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall   : {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.2f}")

# Show decision rules
print("\nDecision Tree Rules:")
print(export_text(model, feature_names=X.columns.tolist()))
