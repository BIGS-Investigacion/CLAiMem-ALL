# =============================================================================
# STATISTICAL COMPARISON: MACENKO VS ORIGINAL
# =============================================================================

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

# Predictions
predictions_original = ...  # (387,)
predictions_macenko = ...   # (387,)
labels_true = ...           # (387,)

# -----------------------------------------------------------------------------
# DESCRIPTIVE STATISTICS + McNEMAR'S TEST
# -----------------------------------------------------------------------------

# Global metrics
acc_orig = accuracy_score(labels_true, predictions_original)
acc_mack = accuracy_score(labels_true, predictions_macenko)

print(f"Global accuracy: {acc_orig:.3f} → {acc_mack:.3f} (Δ={acc_mack-acc_orig:+.3f})")

# Per-class accuracy (for Table 4)
print("\nPer-class accuracy:")
for class_idx, class_name in enumerate(['LumA', 'LumB', 'HER2', 'Basal', 'Normal']):
    mask = (labels_true == class_idx)
    acc_c_orig = accuracy_score(labels_true[mask], predictions_original[mask])
    acc_c_mack = accuracy_score(labels_true[mask], predictions_macenko[mask])
    delta = acc_c_mack - acc_c_orig
    print(f"  {class_name}: {acc_c_orig:.3f} → {acc_c_mack:.3f} (Δ={delta:+.3f})")

# McNemar's test
correct_orig = (predictions_original == labels_true)
correct_mack = (predictions_macenko == labels_true)

contingency = [[
    sum(correct_orig & correct_mack),    # both correct
    sum(correct_orig & ~correct_mack)    # only original correct
], [
    sum(~correct_orig & correct_mack),   # only macenko correct
    sum(~correct_orig & ~correct_mack)   # both wrong
]]

result = mcnemar(contingency, exact=True)

print(f"\n{'='*50}")
print(f"McNEMAR'S TEST")
print(f"{'='*50}")
print(f"χ² = {result.statistic:.3f}")
print(f"p-value = {result.pvalue:.4f}")
print(f"\nConclusion: {'Significant difference' if result.pvalue < 0.05 else 'No significant difference'}")
print(f"Interpretation: Stain normalization {'DOES' if result.pvalue < 0.05 else 'DOES NOT'} significantly improve performance.")