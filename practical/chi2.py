import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Original data
aptitude_scores = [85, 65, 50, 68, 87, 74, 65, 96, 68, 94, 73, 84, 85, 87, 91]
jobprof_scores = [70, 90, 80, 89, 88, 86, 78, 67, 86, 90, 92, 94, 99, 93, 87]

# Convert numerical scores into categories (e.g., High >=85, Medium 70-84, Low <70)
def categorize(score):
    if score >= 85:
        return 'High'
    elif score >= 70:
        return 'Medium'
    else:
        return 'Low'

aptitude_cat = [categorize(score) for score in aptitude_scores]
jobprof_cat = [categorize(score) for score in jobprof_scores]

# Create DataFrame
df = pd.DataFrame({
    'Aptitude': aptitude_cat,
    'Job_Proficiency': jobprof_cat
})

# Create contingency table
contingency_table = pd.crosstab(df['Aptitude'], df['Job_Proficiency'])

print("Contingency Table:")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output results
print("\nChi-square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("P-Value:", p)

alpha = 0.05
if p < alpha:
    print("\nConclusion: Reject Null Hypothesis — There is a significant correlation between aptitude and job proficiency.")
else:
    print("\nConclusion: Fail to Reject Null Hypothesis — No significant correlation between aptitude and job proficiency.")
