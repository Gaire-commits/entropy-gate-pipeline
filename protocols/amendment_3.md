# Amendment 3: a robustness line against a stronger baseline

Dated 2026-11-17, before the reserved period (2024-01-03 to 2026-10-12) was scored, and before PE of daily returns had been compared with a variance baseline on any window. Kshitiz asked for it. It is read together with `protocols/volatility_v2.yaml` and amendments 1 and 2. **STEP 3 stays locked.**

## What changes

One more line in the volatility report's robustness section, which is reported and not decisive. It is the same comparison as the primary one, on the same rows and folds, but against the pooled log-HAR plus the stock's own mean log V over the last 252 sessions (`log_har_long`).

**Nothing else changes:** the baseline stays `log_har`, chosen by its frozen rule on development, and the verdict, critical value and gate are untouched.

## Why

On the official open, `log_har_long` had lower validation losses than the chosen baseline: QLIKE 0.4291 against 0.4346, log loss 0.6301 against 0.6406. If PE stood in for a stock's long-run variance level, which the frozen baseline leaves out, it could pass the primary test for that reason rather than for anything about entropy. This line shows whether a result survives with the long-run level in the baseline.

## How to read it

- **Calibration:** the critical value of 1.94 was calibrated for the frozen baseline, not this one, so this line's Clark-West verdict is indicative only.
- **Survives:** if PE passes against both baselines, the result is not just the long-run level.
- **Doesn't survive:** if PE passes only against the frozen baseline, the primary verdict stands, because that is what was registered. It is then reported with this caveat.

## When it was decided

This was decided after the gate's validation results were visible, which showed baseline losses only and no PE. It was decided before any PE comparison on variance. It cannot change which test is primary.
