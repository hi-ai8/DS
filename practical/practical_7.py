import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Hours_Studied': [2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11],
    'Attendance': [60, 65, 70, 75, 80, 85, 55, 62, 68, 72, 78, 83],
    'Pass': [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]  # Binary output: 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Hours_Studied', 'Attendance']]
y = df['Pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------
# Logistic Regression
# ----------------------
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("🔹 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# ----------------------
# Decision Tree
# ----------------------
tree_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("\n🔹 Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_tree))

# Visualize Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(tree_model, feature_names=['Hours_Studied', 'Attendance'], class_names=['Fail', 'Pass'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
