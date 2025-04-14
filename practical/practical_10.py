import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load Dataset
# We'll use the famous Titanic dataset
titanic = sns.load_dataset('titanic')
print(titanic.head())

# Step 2: Data Preprocessing (drop missing for simplicity)
titanic = titanic.dropna(subset=['age', 'fare', 'sex', 'class', 'survived'])

# Step 3: Create Visualizations

# Plot 1: Distribution of Age
plt.figure(figsize=(8,5))
sns.histplot(titanic['age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Plot 2: Survival count by gender
plt.figure(figsize=(6,4))
sns.countplot(data=titanic, x='sex', hue='survived', palette='Set2')
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.grid(True)
plt.show()

# Plot 3: Fare by Class and Survival
plt.figure(figsize=(8,6))
sns.boxplot(data=titanic, x='class', y='fare', hue='survived', palette='coolwarm')
plt.title('Fare Distribution by Class and Survival')
plt.grid(True)
plt.show()

# Plot 4: Age vs Fare Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=titanic, x='age', y='fare', hue='class', style='survived', palette='deep')
plt.title('Age vs Fare by Class and Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.grid(True)
plt.show()
