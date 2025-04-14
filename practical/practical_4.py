import numpy as np
import pandas as pd
from scipy import stats

# Sample salary data of employees (assumed)
salaries = [48000, 50500, 52000, 49500, 51000, 53000, 47000, 49000, 51000, 51500]

# Step 1: Formulate Hypotheses
# H0 (Null Hypothesis): μ = 50000 (mean salary is ₹50,000)
# H1 (Alternative Hypothesis): μ ≠ 50000 (mean salary is not ₹50,000)

# Step 2: Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(salaries, 50000)

# Step 3: Output results
print("Mean Salary:", np.mean(salaries))
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Step 4: Interpret Results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Average salary is significantly different from ₹50,000.")
else:
    print("Fail to reject the null hypothesis: No significant difference in average salary.")



# Chi-square test example (Goodness of Fit)
from scipy.stats import chisquare

# Observed frequencies
observed = [30, 14, 34, 22]
# Expected frequencies (equal distribution assumed)
expected = [25, 25, 25, 25]

chi_stat, p_val = chisquare(f_obs=observed, f_exp=expected)

print("\nChi-Square Statistic:", chi_stat)
print("P-Value:", p_val)

if p_val < 0.05:
    print("Reject null hypothesis: observed frequencies differ from expected.")
else:
    print("Fail to reject null: no significant difference.")
