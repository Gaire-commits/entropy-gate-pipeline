# Is Clark-West inference appropriate for these models and refits?

Dated 2026-11-02. This is item A2 of `protocols/amendment_1.md`. It was run with `scripts/cw_verification.py`, on data up to the lock, with the reserved period not loaded. No PE of a real return series was set against any target.

**The answer.** For volatility, not as version 1 had it: normal critical values over-reject on this design at every lag tried, so the protocol uses a calibrated critical value of **1.94**, with 21 Newey-West lags. For the portfolio, yes: normal critical values hold with **2** lags, with false alarms a little above nominal in one scenario. In both, averaging across stocks within each date is essential: a stock-day t gives false alarms of 41-45% when the predictor has a slow common swing.

## The two designs

| | Volatility | Portfolio |
|---|---|---|
| Models | Pooled OLS, log-HAR against log-HAR plus PE | Pooled ridge on a rank target, with and without PE's rank |
| Refitting | Recursive, every 21 sessions | Recursive, every 63 sessions |
| Scored | 690 target dates, daily | 98 decisions, every 7 sessions, with targets overlapping for 3 |
| Dependence across stocks | Inside each date's mean | Inside each decision's mean |
| Dependence across dates | Newey-West on that series | Newey-West on that series |

## What the literature settles

[Clark and West (2007)](https://econpapers.repec.org/RePEc:eee:econom:v:138:y:2007:i:1:p:291-311) argue that standard normal critical values give a size close to, and a little under, nominal for nested models. They also concede that the statistic is not asymptotically normal under their preferred conditions, and rolling schemes were generally better sized than recursive ones ([working paper](https://www.nber.org/papers/t0326)). None of that covers a pooled panel with common shocks, refit recursively every few weeks. Ours is that, so the size was measured.

## How it was measured

The test is run many times with a predictor known to carry no incremental information: PE of simulated returns, unrelated to anything real. How often it passes at nominal 5% is its false-alarm rate. Each scenario keeps the same recursive refits as production, and `tests/test_cw_verification.py` checks that the fast engine reproduces the production fit.

**The rule, committed in `7ca6a54` before any of this ran.** The lag to use is the smallest of those tried whose false-alarm rate at nominal 5% is at most 7.5% in every null scenario, simulated and real. If no lag qualifies, normal critical values are not adequate on their own, and a critical value calibrated from placebos replaces 1.645.

### A. Simulated panels (150 stocks, 1,800 sessions, last 690 scored, 500 draws)

Share of draws passing at nominal 5%, in percent. With 500 draws the standard error at these rates is about 1 point.

| Scenario | Lag 5 | Lag 10 | Lag 21 | Lag 42 | Ignoring dependence |
|---|---:|---:|---:|---:|---:|
| Null: independent stocks, normal noise | 4.8 | 4.0 | 3.8 | 3.8 | 4.8 |
| Null: common factor, heavy tails | 8.6 | 7.8 | 6.0 | 5.2 | 9.6 |
| Same, refit every session (not our scheme; 166 draws) | 10.8 | 9.0 | 8.4 | 7.2 | 12.0 |
| Same, rolling 504-session window (not our scheme; 166 draws) | 8.4 | 7.2 | 5.4 | 5.4 | 10.2 |
| Null: the placebo also has a slow common swing | 14.0 | 13.0 | 10.8 | 9.4 | 40.8 |
| True effect, small | 64.4 | 59.4 | 55.0 | 53.4 | 68.8 |
| True effect, moderate | 96.4 | 96.0 | 95.2 | 93.6 | 98.4 |

Our refitting scheme is not the problem: 21-session recursive refits sit between refitting every session and a rolling window. The moderate true effect also shows why Clark-West and the raw loss are kept apart (A1): it passes Clark-West in 95% of draws, and in 41% of draws it does so with the raw loss not significantly lower.

### B. The real target with placebos (755,685 rows, last 690 target dates before the lock, 500 draws)

| Scenario | Lag 5 | Lag 10 | Lag 21 | Lag 42 | Ignoring dependence |
|---|---:|---:|---:|---:|---:|
| Null: the placebo's own swings only | 7.8 | 7.8 | 7.4 | 8.0 | 7.8 |
| Null: a slow common swing | 11.6 | 10.0 | 9.6 | 10.4 | 44.6 |

### C. The portfolio (the last 98 decisions before the lock, 500 draws; lags in decisions)

| Scenario | Lag 2 | Lag 4 | Lag 9 | Lag 18 | Ignoring dependence |
|---|---:|---:|---:|---:|---:|
| Null: the placebo rank's own swings only | 6.8 | 7.0 | 7.4 | 8.4 | 10.0 |
| Null: a slow change in which stocks rank high | 3.8 | 3.8 | 4.4 | 6.0 | 6.8 |

## The rule applied

**Portfolio:** lag 2 is the smallest that qualifies, at 6.8% and 3.8%. It is also the fewest that cover the target's overlap of 3 decisions. 1.645 stands. Its 2.5% value for Holm is the normal 1.96, which was not measured.

**Volatility:** no lag qualifies, on simulated panels or the real target, so a calibrated critical value replaces 1.645.

## How the critical value was calibrated

The rule did not say how, so this was fixed and committed in `caa83bb` before it ran. By then parts A and the description below had been seen.

### P. What the real predictor looks like (up to the lock, with the target dropped before anything is computed)

Part A's swing was a stress test. The calibration should resemble the real predictor, so PE's own movement was measured. This says nothing about forecasting.

| | Standard deviation | Of the date mean | Common share of variance | Of the stock mean | Date-mean autocorrelation, 1 / 21 / 63 sessions |
|---|---:|---:|---:|---:|---:|
| Real PE, up to the lock | 0.0128 | 0.0011 | 0.7% | 0.0020 | 0.96 / 0.45 / 0.03 |
| Development | 0.0128 | 0.0010 | 0.6% | 0.0024 | 0.96 / 0.37 / -0.01 |
| Validation | 0.0127 | 0.0013 | 1.0% | 0.0037 | 0.97 / 0.56 / 0.06 |
| Placebo, own swings only | 0.0128 | 0.0006 | 0.2% | 0.0020 | 0.96 / 0.35 / -0.01 |
| **Placebo, matched to real PE** | 0.0129 | 0.0013 | 1.0% | 0.0020 | 0.97 / 0.41 / 0.02 |
| Placebo, adverse swing | 0.0182 | 0.0129 | 50% | 0.0021 | 0.97 / 0.42 / 0.03 |

Real PE moves together about twice as much as independent noise would in standard deviation, and about a tenth as much as the adverse swing. (Parts A and B do not rescale the swing per draw, so their share runs near 40%. Part D rescales it to exactly PE's own standard deviation, so 50%.) Its stock means vary no more than a placebo's across the whole sample, so it has little lasting stock-level component.

### D. The calibration (real target, recursive, refit every 21 sessions)

**The rule.** The 95th percentile of the Clark-West t at 21 Newey-West lags, the refit cycle and the value set before part A, over placebos matched to real PE at its larger date-mean value across development and validation. Taken on two windows of 690 target dates, the larger of the two, rounded up to two decimals. The 97.5th, the same way, serves Holm. The other two scenarios are reported and not used.

| | Last 690 target dates (2021-03-05 to 2024-01-02) | | | The 690 before those (2018-05-25 to 2021-03-04, with March 2020) | | |
|---|---:|---:|---:|---:|---:|---:|
| | **Matched** | Own swings | Adverse | **Matched** | Own swings | Adverse |
| Draws | 4,000 | 2,000 | 2,000 | 4,000 | 2,000 | 2,000 |
| False alarms at 1.645, lag 5 | 9.2% | 6.2% | 13.1% | 11.3% | 5.0% | 17.1% |
| Lag 10 | 8.5% | 6.3% | 11.9% | 8.8% | 4.3% | 13.3% |
| Lag 21 | **8.3%** | 5.9% | 11.3% | **7.0%** | 3.3% | 10.3% |
| Lag 42 | 8.3% | 5.9% | 10.9% | 6.2% | 3.0% | 9.1% |
| 95th percentile, lag 21 | **1.94** | 1.68 | 2.12 | **1.84** | 1.41 | 2.03 |
| 97.5th percentile, lag 21 | **2.33** | 2.12 | 2.45 | **2.15** | 1.77 | 2.34 |
| False alarm of the full verdict at its own 95th: clears it and raw loss fell | 1.3% | 1.1% | 1.3% | 0.8% | 0.3% | 1.5% |

**The result: 1.94 at 5%, 2.33 at 2.5%.** With 4,000 draws, each is good to about ±0.1 at 95% confidence. The rule's branch does not hinge on the adverse scenario: on the matched placebo, 1.645 at 21 lags still gives 8.3% on the later window.

If the adverse swing had been used instead, the critical value would be 2.12. If no common movement had been assumed, 1.68. The excess false alarms rise with the amount of common movement in the predictor, which is why the calibration follows the measured amount.

## What it costs: power at 1.94

On the real target, over the last 690 target dates before the lock, with a true effect of known size planted on the matched placebo. It is planted in the target and in the inputs, as a real one would be, because a slow-moving PE's effect also sits in past variance and the baseline absorbs much of it. 1,000 draws each. No rule rests on this.

| If one standard deviation of PE moved next-day variance by | Mean fall in log loss | Clears 1.94 | Clears, and raw loss significantly lower | Mean t |
|---:|---:|---:|---:|---:|
| 1.3% | under 0.0001 | 35% | 11% | 1.3 |
| 2.6% | 0.0001 | 82% | 44% | 3.3 |
| 6.6% | 0.0004 | 100% | 95% | 8.0 |
| 13.7% | 0.0015 | 100% | 100% | 13.6 |

So a null would rule out an effect near 7% per standard deviation and say little about one near 1%. A pass could also be small: the 6.6% row lowers a log loss of about 0.64 by 0.06%, which is why amounts are reported beside the test. "Significantly lower" here is a Newey-West t of at least 1.96 at 21 lags, standing in for the bootstrap interval the verdict uses.

A first version of this check left the inputs unchanged and gave 99% in the 1.3% row. That overstated power several times over, and it was caught because it disagreed with part A.

## What this does not settle

- **Placebos are not real PE.** They share its size and common movement, not any link it may have with a stock's own return behaviour. A truly null real PE cannot be simulated exactly.
- **The reserved dates' dependence can differ.** The calibration uses the target up to the lock, on the vendor's open. The earlier window includes March 2020, and it was the later window that set the value, but April 2025 is outside both.
- **A scenario that tested nothing.** Part A's "placebo correlated with the inputs" added a multiple of an input to the placebo. That leaves the fit unchanged, so its numbers matched the previous row exactly in part A, and in part B differed only by the random draws (9.6% against 10.0% at 21 lags), which is a useful picture of the sampling noise. It is left out of the tables above. A predictor correlated with something the baseline omits is not a null at all: it has real incremental content.
- **Passing says content over this baseline, not why.** The baseline is the rule's pick, not the best model possible: adding the stock's long-run level has the lower validation losses. A significant PE could reflect what the baseline leaves out rather than entropy itself. Part P's stock means make that less likely for the long-run level, but do not rule out other routes.
- **Portfolio false alarms are not at nominal.** At lag 2 they are 6.8% in one scenario, accepted by the rule's 7.5% limit, and it did not ask for more. The verdict there also needs the raw loss and the IC to agree: at 9 lags, passing Clark-West with the raw loss also significantly lower happened in 1.0% and 0.6% of draws.

## To rerun

```bash
python scripts/cw_verification.py --sessions sessions.parquet --reference data/reference --reps 500 --parts abc
python scripts/cw_verification.py --sessions sessions.parquet --reference data/reference --reps 4000 --parts pd --workers 7 --arrays cw_real_target.npz
python scripts/cw_verification.py --sessions sessions.parquet --reference data/reference --reps 1000 --parts e --critical 1.94 --workers 8 --arrays cw_real_target.npz
```
