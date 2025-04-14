import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

# Sample data for four groups
group1 = [23, 25, 29, 34, 30]
group2 = [19, 20, 22, 25, 24]
group3 = [15, 18, 20, 21, 17]
group4 = [28, 24, 26, 30, 29]

# Combine data and create labels
all_data = np.concatenate([group1, group2, group3, group4])
group_labels = ['G1']*5 + ['G2']*5 + ['G3']*5 + ['G4']*5

# One-Way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3, group4)
print(f"ANOVA: F={f_stat:.4f}, p={p_value:.4f}")

# If significant, perform post-hoc test
if p_value < 0.05:
    print("Result: Groups have different means")
    # Tukey's test to find which groups differ
    tukey = pairwise_tukeyhsd(all_data, group_labels)
    print(tukey)
else:
    print("Result: No significant difference between groups")