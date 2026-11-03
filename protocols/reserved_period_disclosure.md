# Prior use of the reserved period, 2024-01-03 to 2026-10-12

Written 2026-11-02, before the period was scored under the protocols, and updated the same day for their second version (`protocols/volatility_v2.yaml`, `protocols/portfolio_v2.yaml`, by `protocols/amendment_1.md`). To be cited with either result.

**The period is reserved for these specifications but previously exposed through related research. It is not an untouched test.** Walk-forward fitting stops a model from seeing the future. It cannot stop earlier findings on these same dates from having shaped the questions, features and settings. Only trading days after 2026-11-02 can be untouched; the protocols do not use them.

Sources: `RESULTS.md`, `HANDOVER.md`, the configs named below, and `~/Transfer-entropy-` (`README.md`, `configs/heat_v1.yaml`, `configs/heat_v2.yaml`). Where the record is silent, this says so.

## What has been scored on these dates

| Work | Run | Data | What was scored on the reserved dates | PE involved? |
|---|---|---:|---|---|
| **HEAT1** (registered), transfer-entropy repo | 2026-10-22 | 5-minute SIP, 503 stocks | **Its test window is this period**: 2024-01-03 to 2026-10-12 (last day inferred), 3.29 million moments, 30 minutes to 4 hours. 0 of 12 cells with an edge | No |
| HEAT macro-day split | 2026-10-24 | Saved HEAT1 test predictions | Exploratory split of the same test days by jobs, CPI, GDP and FOMC days | No |
| **HEAT2** (registered 2026-10-23) | **Not recorded** | Same split | The repo holds only a check run on 2021 (`heat_v2_check_results.zip`, "the registered test days were not used"). Whether the registered test ran is **unknown**; asked twice, unanswered. If it ran, it is a further look at this period | No |
| SIP sweep without entropy (`sip_pe_free`) | 2026-10-27 | 5-minute SIP, 503 stocks | Walk-forward test folds 2018-04 to 2026-10: return at 2.5 hours. Does not beat cost | No |
| `gbm_ent` on the same rows (`sip_sp500_multihour`) | 2026-10-27 | Same | Same folds, with 5-minute entropy features. Entropy adds nothing | **Yes, 5-minute PE, for returns** |
| E5 lead-lag, E7-E10 reversal studies, E12 peers | 2026-10-13 and 14 | 5-minute SIP, 2016 to 2026-10 | Short-term reversal, its size, horizons, holds; correlated peers. The reversal was found here (IC +0.0064) | No |
| **E6** reversal by entropy | 2026-10-14 | 5-minute SIP, 2016 to 2026-10 | Reversal by thirds of **5-minute PE**, and by thirds of volatility. PE does not condition it; volatility does (reported, not decisive) | **Yes, 5-minute PE against volatility** |
| E11 mid-cap replication | 2026-10-14 | 5-minute SIP, 399 mid caps | Same dates, other stocks | No |
| E1 volatile moments | 2026-10-08 and 13 | IEX from 2020-07-28; then SIP | Volatility as a trading selector, including April 2025. Not replicated | Volatility, not PE |
| Forward-test power (section 30) | 2026-10 | SIP, both universes | A design calculation on 2016 to 2026 | No |
| IEX sweeps (`spy_gao`, `etf_gao`, `etf_intraday`, `sp500_multihour`), gate tests | to 2026-10 | IEX | 15 walk-forward folds, late 2022 to August 2026 | **Yes, 5-minute PE: gate and inputs** |
| Selective trading, breadth, decision engine | 2026-10 | IEX | Reused those predictions | **Yes, 5-minute PE: inputs** |
| Nasdaq-100 hold-out and replication, online agent, E2 | 2026-10 | IEX, with test periods reaching 2026 | Later quarters on 2020-2026 data | Partly, as model inputs |

**HEAT1 also tuned on validation** (2022-01-04 to 2023-12-29, on 2026-10-22), the same dates as validation here.

## What this project itself has seen in the period, with no forecast involved

Part of the data audit for these protocols ran over the whole span, so the following are known:

- **Levels and regimes.** Market-wide variance by year, including 2024, 2025 and 2026 (typical daily vol 1.96%, 2.33%, 2.54%); mean squared overnight gap by year; the highest session of the window (2025-04-09); the distribution of V over all sessions (`RESULTS.md`, "The volatility panel").
- **Named extreme events.** Large moves were listed and classified, among them GL 2024-04-11, FISV 2025-10-29 and MRNA 2026-08-19. MRNA was checked against Yahoo's bars and news on 2026-11-02: a real +177%, on Moderna and Merck's Phase 3 INTerpath-001 result. These stay in the data.
- **The open.** Section 38 printed, by year including 2024-2026, on 20 stocks: how far the vendor's open is from the opening print, the mean of an approximate V, and how V would change. It did not relate V to PE or to any forecast.
- **Counts.** Eligible stock-days per window.
- **Portfolio tool, 2026-11-02.** The first version listed the largest moves in the data, including those above, and one list of stocks the PE-free model would hold on 2026-10-13. It printed no returns after 2024-01-02 and scored nothing in the period. Later versions load nothing past the lock.
- **Yahoo check, 2026-11-02.** Requests for SPY, BIL and two index series returned one bar dated 2024-01-03, which was cut unread.
- **From memory,** not measured, in the notes: a calm 2024 with an August spike, the April 2025 tariff shock, a rising 2026.

From amendment 1 on, the checks print only coverage counts for this period, and no return or open-gap summary of it. The items above were seen before that rule.

## What was done up to the lock while amending, 2026-11-02

Nothing here loaded a date after 2024-01-02 for analysis or scored the period, but each could have shaped the protocols:

- **The baseline search ran again** on development and validation, with no PE, on the vendor's open, under the amended conversion and liquidity measure. Its numbers were visible, and the first search's were visible when the amendment was written.
- **The Clark-West verification** (`protocols/clark_west_verification.md`) used the real target up to the lock against placebo predictors built from simulated returns, and the portfolio's target the same way. No PE of a real return series was set against any target.
- **A description of real PE up to the lock,** with the target dropped first: how much PE moves together across stocks, and how slowly. It chose the placebo behind the volatility protocol's critical value. It says nothing about forecasting.
- **The portfolio's locked report ran again** on development and validation, including PE against `base` there. `base_pe` passes on neither window.

## What has not been done

- **PE of daily returns against next-day variance has not been compared on any window** of this data, development and validation included.
- **PE of daily returns against returns** has been compared on development and validation only (portfolio tool, 2026-11-02, on successive sets of rules), not on the reserved period. No setting or rule was chosen from that comparison, but the PE settings were reused for volatility with it visible (`protocols/amendment_1.md`, A6).
- **The final portfolio rules** were chosen on the PE-free model on development and validation only, as the best of 24 candidates seen across two searches; the second search's design followed from reading the first's results (`protocols/portfolio_rule_search_v2.yaml`).
- **The volatility panel on the official open** has not run for all stocks, on any window: it needs Alpaca's prints. Numbers in the record so far use the vendor's open as a stand-in.

## How to read a result from this period

1. Say which protocol it comes from and quote the first paragraph above.
2. A null is informative: earlier work on these dates found little, and nothing was tuned to it.
3. A positive result is weaker than an untouched one. The confirmation it would need is a forward test on trading days after 2026-11-02, with the rule unchanged.
4. Two primary tests share this period, on different targets (next-session variance; 21-session returns). Report each verdict, plus a Holm correction across both.
5. A significant Clark-West result is not a fall in out-of-sample loss. Quote the unadjusted loss difference and its interval beside it, and QLIKE.
