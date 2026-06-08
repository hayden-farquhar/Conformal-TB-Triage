"""
Held-out-calibration Venn-ABERS probability intervals.

The original implementation fit the Venn-ABERS isotonic calibrators on the
probe-TRAINING split (Shenzhen+Montgomery 'calibration') -- the same in-sample
defect this revision addresses for the conformal sets. Intervals fit on the
predictor's own training data are not out-of-sample and understate uncertainty.

This version fits the two isotonic calibrators on the held-out TBX11K `dev`
split (out-of-sample for the NLM-trained probe, in-distribution with test) and
applies them to the fixed test set. The probe is unchanged; only the provenance
of the calibration scores changes. CPU only -- operates on the saved fixed-probe
predictions (outputs/probe_predictions.parquet), no embedding re-run.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs"
PRED_PATH = RESULTS_DIR / "probe_predictions.parquet"
TABLES_DIR = RESULTS_DIR / "tables"
EMB, PROBE = "rad_dino", "linear"


def load_dev_test():
    d = pd.read_parquet(PRED_PATH)
    d = d[(d.embedding == EMB) & (d.probe == PROBE)]
    dev, test = d[d.split == "dev"], d[d.split == "test"]
    return (dev.y_prob.to_numpy(), dev.y_true.to_numpy().astype(int),
            test.y_prob.to_numpy(), test.y_true.to_numpy().astype(int))


def main():
    cal_scores, cal_y, test_scores, test_y = load_dev_test()

    base = np.append(cal_scores, test_scores)
    y0 = np.append(cal_y, np.zeros(len(test_scores)))
    y1 = np.append(cal_y, np.ones(len(test_scores)))

    iso0 = IsotonicRegression(out_of_bounds="clip").fit(base, y0)
    iso1 = IsotonicRegression(out_of_bounds="clip").fit(base, y1)
    p0, p1 = iso0.predict(test_scores), iso1.predict(test_scores)

    lower, upper = np.minimum(p0, p1), np.maximum(p0, p1)
    widths = upper - lower
    decisive_in = (lower > 0.50).mean()
    decisive_out = (upper < 0.10).mean()
    tb_lower_pos = (lower[test_y == 1] > 0).mean()

    print(f"Venn-ABERS (held-out dev calibration, n_cal={len(cal_y)}):")
    print(f"  Mean interval width: {widths.mean():.4f}")
    print(f"  Total decisive: {decisive_in + decisive_out:.4f}")

    pd.DataFrame({"patient_id": range(len(lower)), "p_lower": lower,
                  "p_upper": upper, "width": widths, "y_true": test_y}
                 ).to_csv(TABLES_DIR / "venn_abers.csv", index=False)
    pd.DataFrame([{"n_cal": len(cal_y), "mean_width": round(widths.mean(), 4),
                   "median_width": round(float(np.median(widths)), 4),
                   "decisive_rule_in": round(decisive_in, 4),
                   "decisive_rule_out": round(decisive_out, 4),
                   "decisive_total": round(decisive_in + decisive_out, 4),
                   "tb_lower_pos": round(tb_lower_pos, 4)}]
                 ).to_csv(TABLES_DIR / "venn_abers_summary.csv", index=False)
    print("Wrote venn_abers.csv (+_summary).")


if __name__ == "__main__":
    main()
