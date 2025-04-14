import scipy.stats as stats

# Sample data and population mean
population_mean = 50
sample_data = [45, 55, 52, 48, 49, 51, 53, 47, 54, 50]

# One-Sample t-test
t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)

# Interpret results with alpha = 0.05
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print("Conclusion:", "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis")


#?custom
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Step 1: Raw data for aptitude and job proficiency
aptitude = [85, 65, 50, 68, 87, 74, 65, 96, 68, 94, 73, 84, 85, 87, 91]
jobprof = [70, 90, 80, 89, 88, 86, 78, 67, 86, 90, 92, 94, 99, 93, 87]

# Step 2: Convert continuous scores to categorical levels
aptitude_cat = pd.cut(aptitude, bins=[0, 70, 85, 100], labels=["Low", "Medium", "High"])
jobprof_cat = pd.cut(jobprof, bins=[0, 80, 90, 100], labels=["Low", "Medium", "High"])

# Step 3: Create a contingency table
observed = pd.crosstab(aptitude_cat, jobprof_cat)

# Step 4: Perform Chi-Square Test
chi2_stat, p_value, dof, expected = chi2_contingency(observed)

# Step 5: Display results
print("\n**Chi-Square Test Results**")
print(f"Chi-Square Statistic: {chi2_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:\n", expected)

# Step 6: Decision Rule
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Conclusion: Reject H0 → Aptitude and Job Proficiency are dependent.")
else:
    print("Conclusion: Fail to reject H0 → No significant relationship between Aptitude and Job Proficiency.")


#!!chi-suare
import scipy.stats as stats
import numpy as np

# Contingency table [Male, Female] x [Success, Failure]
observed = np.array([[40, 30], [35, 25]])

# Chi-Square Test for Independence
chi2_stat, p_value, _, _ = stats.chi2_contingency(observed)

# Results and conclusion
print(f"Chi-Square: {chi2_stat:.4f}, p-value: {p_value:.4f}")
print("Conclusion:", "Dependent variables" if p_value < 0.05 else "Independent variables")