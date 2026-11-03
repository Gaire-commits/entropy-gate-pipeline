# Amendment 1 to the portfolio and volatility protocols

Dated 2026-11-02. Requested by Kshitiz on review of `protocols/volatility_v1.yaml` and `protocols/portfolio_v1.yaml`, before the reserved period (2024-01-03 to 2026-10-12) was scored under either, and before PE of daily returns had been compared with a variance baseline on any window. It produces `protocols/volatility_v2.yaml` and `protocols/portfolio_v2.yaml`. The version 1 files stay as they were, and the code they ran on is at commit `8ee50cb`.

**Retained, on Kshitiz's decision:** the PE specification, the explicitly defined V proxy, the half-day treatment, and index-join dates with the survivorship limitation clearly stated. **STEP 3 stays locked:** nothing here scores the PE comparison on the reserved period, and neither protocol allows that without Kshitiz's word.

The code came first, in commit `7ca6a54`, with the Clark-West verification's lag rule committed before that verification ran.

## A1. A significant Clark-West result is not a fall in out-of-sample loss

**The problem.** Clark-West adds the squared gap between the two forecasts to the raw loss difference, to allow for the noise the extra parameter adds when it is truly zero. So it asks whether the larger model has incremental predictive content. It can pass while the larger model's errors in the sample are no smaller. Version 1 said "Clark-West t >= 1.645: PE improves the forecast", which ran the two together.

**The rule now.** Three things are reported apart: Clark-West; the unadjusted paired log-loss difference with its 95% block-bootstrap interval and Newey-West t; and QLIKE the same way. The verdict is one of:

| Clark-West t clears its critical value (A2) | Log-loss interval wholly below zero | Verdict |
|---|---|---|
| Yes | Yes | PE improves the forecast of log variance out of sample, and QLIKE agrees or does not confirm it |
| Yes | No | Clark-West detects incremental content, but out-of-sample log loss did not demonstrably fall, and QLIKE did or did not fall |
| No | Either | No improvement shown |

The portfolio's forecast verdict works the same way, on the target's rank, and also needs the IC gap's interval wholly above zero. Its net-return verdict is unchanged. Code: `src/volforecast.py` (`gap`, `clark_west`, `pe_test`, `verdict`) and `src/portfolio.py` (`forecast_losses`, `paired_comparison`, `verdict`).

## A2. Whether Clark-West inference suits the pooled models and their refitting

**The problem.** Version 1 took the Clark-West t, with 5 Newey-West lags for volatility and 4 for the portfolio, against 1.645. The literature does not cover a pooled panel refit recursively every few weeks, so this was measured. `protocols/clark_west_verification.md` has the method and the tables.

**The rule, committed in `7ca6a54` before the measuring ran.** Use the smallest lag tried whose false alarms at nominal 5% are at most 7.5% in every null scenario, simulated and real. If none qualifies, a critical value calibrated from placebos replaces 1.645.

**What it gave.**

| | Newey-West lags | Critical value, 5% / 2.5% (for Holm) | How |
|---|---:|---:|---|
| **Volatility** | 21, in target dates | **1.94 / 2.33** | No lag qualified, so calibrated from placebos matched to real PE's common movement, on the real target up to the lock. The way to calibrate was committed in `caa83bb` before it ran |
| **Portfolio** | 2, in decisions | 1.645 / 1.96 | Lag 2 qualified, with false alarms of 6.8% and 3.8%. It is also the fewest lags that cover the target's overlap of 3 decisions |

Dependence across stocks is inside each date's mean, which matters most of all: a stock-day t would give false alarms of 41-45% when the predictor has a slow common swing. The refitting scheme itself was not the problem: on a simulated panel with common factors, 21-session recursive refits gave 6.0% at 21 lags, and a rolling 504-session window 5.4%. The excess came from slow common movement in the predictor, and even real PE's small amount is enough: on the real target at 21 lags and 1.645, false alarms were 5.9% and 3.3% on the two windows with none, 8.3% and 7.0% with real PE's amount, and 11.3% and 10.3% with the adverse swing. With 1.94, the full verdict "PE improves the forecast", which also needs the raw loss to fall, false-alarms about 1% of the time on those placebos.

## A3. From log forecasts to variance forecasts, for QLIKE

**The problem.** QLIKE needs a variance, and the models forecast its logarithm. Version 1 used `exp(f + s²/2)`, which is right only if the residuals are normal, and they are not.

**The rule now.** `h = exp(f) × s`, with `s` Duan's (1983) smearing factor: the mean of `exp(residual)` over that refit's own training rows. Those are rows whose target was realized by the refit date, so nothing later enters. Each model and each refit has its own factor, and the factor needs no assumption about the residuals' shape. The log loss does not depend on it. Each report prints the mean realized variance over the mean forecast variance as a check. Code: `src/volforecast.py` (`_ols`, `walk_forward`, `calibration`).

**What it did, on the vendor's open.** The form is reselected by the same rule (`protocols/vol_baseline_search.yaml`), and the rule's pick does not change:

| | Development QLIKE, first search → amended | Validation QLIKE, first search → amended | Realized over forecast variance, development / validation |
|---|---:|---:|---:|
| **pooled log-HAR (the rule's pick both times)** | **0.4854 → 0.4682** | 0.4426 → 0.4330 | 1.107 / 0.960 |
| plus the stock's long-run level | 0.4971 → 0.4749 | 0.4328 → 0.4276 | 1.134 / 0.924 |
| log-HAR per stock | 0.4985 → 0.4776 | 0.4388 → 0.4282 | 1.128 / 0.956 |
| level HAR, which needs no conversion | 0.5325 → 0.5324 | 0.4473 → 0.4471 | 1.088 / 0.988 |
| monthly mean carried forward, which needs none | 0.5617 → 0.5616 | 0.4909 → 0.4908 | 1.002 / 1.001 |

The second figures also include A4's liquidity measure, whose effect shows in the two rows that need no conversion. The gate passes: against the monthly mean on validation, QLIKE gap -0.0577 [-0.0782, -0.0376] and log loss gap -0.2002 [-0.2568, -0.1417]. The mean smearing factor is 1.588, where a normal assumption at this residual variance would give about 1.4. Mean realized variance is 11% above the mean forecast on development and 4% below it on validation.

These numbers came after the first baseline search's, which were visible when this conversion was chosen. PE is in none of them.

## A4. The missing official open, and one liquidity measure

**Missing official open.** Version 1 fell back on the vendor's open, flagged. Now V is missing for that session, with the reason `no official open`, and nothing is substituted. The session cannot be a target and counts against the stock's validity rules; the range alone is unaffected. The vendor's open in those places is a reported sensitivity only. Since this removes data, it comes with two checks that score nothing: among usable bars at least 95% must have an official open in every year, and before the reserved period sessions in the top 1% by V may lack one at most max(2 × the overall share, 2%) of the time. The second replaces nothing in version 1; it is here because a policy that drops sessions could otherwise drop the largest moves. Code: `src/volpanel.py` (`build_panel`), `src/official.py` (`eligible_coverage`, `missing_on_extreme_days`), `scripts/protocol_checks.py`.

**Liquidity.** Version 1 had two measures under one name: in-session dollar volume from the 5-minute cache for volatility, and the vendor's daily dollar volume for the portfolio. Both now use the median of quoted close × the vendor's raw daily volume over the 63 earlier sessions, with at least 40 available, through one function (`volpanel.trailing_dollar_volume`). That volume includes extended hours, about 28% above the session's, so the $10M rule is slightly lenient, and both protocols say so. The threshold does not change. Through the lock, this adds 591 development rows and 1,240 validation rows on the vendor's open, and one stock in validation (451 against 450).

## A5. No reserved-period summaries during the checks

| Check | For the reserved period it prints | It no longer prints |
|---|---|---|
| `scripts/official_prices.py` | Bars, listing and official-print counts and shares by year; which symbols lack the most opens | Open or close gaps, in size or share. `official.gap_summary` computes nothing on or after the date it is given |
| `scripts/protocol_checks.py` | Official-open coverage among usable bars by year | Benchmark returns (fetched only through 2024-01-02); the clustering check stops before the period; spreads are sampled from 2019 to 2023 |
| `scripts/build_sessions.py`, new, in place of the full audit for this purpose | Nothing: `--until 2024-01-02` reads no later bar | The audit's yearly variance, overnight and event tables |
| `scripts/vol_forecast.py --stage baseline` | Nothing: nothing after the lock is loaded | The reserved row count, which is not recounted under these rules |

The checks' data files still hold those dates, because coverage needs them and scoring would later. Nothing summarizes them.

## A6. The protocol history, corrected

Version 1's list of "choices made in freezing" put these together, and its phrase "fixed in advance" for the PE settings was too loose. The record, by commit:

### Before any daily forecast (2026-10-27 and 28)

| Choice | Source | Status then |
|---|---|---|
| The question, one PE hypothesis, nested comparison with Clark-West on identical folds, positive or null accepted | Kshitiz's brief | Fixed |
| A range proxy with the overnight explicit as primary; realized variance as robustness | Kshitiz's brief | Fixed |
| V's formula, with V_id as a sensitivity | Data audit, `8846313` | Proposed |
| Half-days as sessions of their own, by the exchange calendar | Data audit, on Kshitiz's instruction | Fixed. Whether to score half-day targets stayed open |
| The official open | Measured in `f2ecce0` on 20 stocks, which printed open gaps by year including 2024-2026 | Proposed |
| The $5 and $10M thresholds; HEAT1's window boundaries | `8846313` | Proposed |

The same record lists as undecided: the PE settings, the HAR form, the loss, and half-day targets.

### After the first portfolio results (2026-11-02, from `03ca3c3`)

Those results included PE against `base` on development and validation, and showed nothing the rule could tell from zero.

| Choice | Protocol | Commit | What was visible |
|---|---|---|---|
| The rule search and its amendment to staggered tranches; the final rules | Portfolio | `fe44480`, `a1270d2` | The PE-free model's results, 24 candidates in all. The amendment followed from reading the first search |
| Index-join dates | Portfolio | `a1270d2` | The portfolio's diagnostics: equal-weight's return was inflated |
| The target's match to next-close fills, and the physical lock | Portfolio | `fe44480` | The first version's results |
| Leaving half-day targets out; the log loss and floor; the HAR candidates; the QLIKE rule and gate | Volatility | `52a750b` | Portfolio results only. No variance forecast existed yet |

The portfolio's costs and features were in `03ca3c3`, before its first results, and have not changed. The check against published trading costs came later and changed nothing.

### Settings reused for volatility, with those portfolio results visible

| Setting | Written for | Reused in | What that means |
|---|---|---|---|
| PE order 3, delay 1, 63 patterns, at least 57 valid | The portfolio tool, `03ca3c3`, before its first run | Volatility, `52a750b` | Never tuned on any result, but not chosen for variance either. "Fixed in advance" holds only against variance forecasts. Version 1's reason, "so the shared period has one PE", was a convenience, not evidence. The two protocols also enter PE differently: its level for volatility, its rank for the portfolio |
| Index-join dates | The portfolio, after its diagnostics | Volatility, `52a750b`, before any variance forecast | The reason came from the portfolio's results and applies to both |
| Window boundaries | HEAT1, 2026-10-22 | Both | Shared with a study on other questions. See the disclosure |

Reuse also ran the other way: the $5 and $10M thresholds came from the volatility panel, and the portfolio tool took them over. Now the liquidity measure comes from the portfolio and volatility takes it over.

### After the baseline search's results, before any PE comparison on variance (2026-11-02)

This amendment: the conversion, the missing-open policy, the shared liquidity measure, the Newey-West lags and critical values, and the verdict. The baseline search's numbers, on the vendor's open with no PE, were visible throughout. None of these choices could have used a result of PE against variance, because none exists. The critical value used a description of real PE with the target dropped. On review, Kshitiz retained the PE specification, V, the half-day treatment and the index-join dates.

## Also changed, beyond the six items

These are consequences of the items above, chosen by Claude. Kshitiz can replace them by a dated amendment before anything is scored.

1. **The form must survive the official open.** Version 1 kept the form regardless. Now `--stage baseline` reselects it on the registered target, and if the rule picks another form the evaluation stops for an amendment. It cannot favour PE, since PE is not in that search.
2. **Holm is now concrete.** Version 1 said a Holm correction would be reported and nothing computed one. With a calibrated critical value a normal p-value no longer fits, so each protocol fixes a 5% and a 2.5% critical value and each report says which its Clark-West t clears. The rule: a test clearing its own 2.5% value passes, and the other then needs only its own 5% value. Verdicts remain the unadjusted ones.
3. **STEP 2 and STEP 3 of `notebooks/reserved_evaluation.ipynb` no longer run the full audit,** which prints every year. They use `scripts/build_sessions.py`, which reads the 5-minute cache only up to the date given.
4. **The calibration's placebo.** The rule committed first said only "calibrated from placebos". Choosing the placebo matched to real PE gave 1.94. The adverse swing would give 2.12, and no common movement 1.68.
5. **The clustering check on missing opens** (A4), which version 1 did not have.
