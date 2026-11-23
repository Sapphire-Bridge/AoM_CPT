"""Verify Appendix F using the archived CF patching CSV. No model is needed.

Run with: python3 check_appendix_f.py /path/to/cf_patching_shift_vs_subinv.csv
Requires NumPy, which the original percentile bootstrap also uses.
"""

import argparse
import csv
import math

import numpy as np


def verify(row):
    shift = "cf_patch_stratum_expected_effect__shift_mean_max_effect"
    control = "cf_patch_stratum_expected_effect__invariant_mean_max_effect"
    noun = "cf_patch_stratum_intervention_type__substitution_noun_synonym_mean_max_effect"
    if tuple(int(row[key + "_n"]) for key in (shift, control, noun)) != (60, 20, 3):
        raise ValueError("Expected 60 shift items, 20 controls and 3 noun substitutions.")
    n_boot, seed, ci = int(row["bootstrap_n"]), int(row["bootstrap_seed"]), float(row["ci"])
    if (n_boot, seed, ci) != (1000, 42, 0.95):
        raise ValueError("Unexpected bootstrap settings for this archive.")

    # With three cases, a resample consisting of three copies of the same
    # case has exactly that case's mean. Each index occurs this way often
    # enough to cover the percentile endpoints, in any ordering of the cases.
    draws = np.random.RandomState(seed).randint(0, 3, size=(n_boot, 3))
    counts = [int(np.all(draws == i, axis=1).sum()) for i in range(3)]
    if min(counts) <= math.ceil(0.025 * (n_boot - 1)):
        raise ValueError("Bootstrap bounds do not guarantee the observed extremes.")
    low, high = float(row[noun + "_ci_low"]), float(row[noun + "_ci_high"])
    noun_mean = float(row[noun])
    noun_values = (low, 3 * noun_mean - low - high, high)

    shift_mean, control_mean = float(row[shift]), float(row[control])
    d_all = float(row["cf_patch_comparison_expected_effect_shift_vs_invariant_cohens_d"])
    sd_all = (shift_mean - control_mean) / d_all
    # 60+20-2 degrees of freedom in the original pooled sum of squares.
    ss_all = 78 * sd_all**2
    kept_mean = (20 * control_mean - 3 * noun_mean) / 17
    # Remove the three squared deviations about the original control mean.
    # Recenter the 17 remaining controls on their own mean.
    ss_kept = ss_all - sum((v - control_mean) ** 2 for v in noun_values)
    ss_kept -= 17 * (kept_mean - control_mean) ** 2
    sd_kept = math.sqrt(ss_kept / 75)  # 60+17-2
    difference = shift_mean - kept_mean
    return row["model"], difference, sd_all, sd_kept, difference / sd_all, difference / sd_kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args()
    print("model                mean diff   SD(60:20)   SD(60:17)   d(old SD)   d(60:17)")
    with open(args.csv_path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            model, *values = verify(row)
            print(f"{model:20}" + "".join(f"{value:12.6f}" for value in values))
