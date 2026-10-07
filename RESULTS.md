# Results and progress

Last updated 2026-10-07. Numbers come from each experiment's `outputs/<experiment>/summary.md` and
`engine.md`, which live on Google Drive rather than in this repo. The runs finished between late September
and 7 October 2026.

## In short

- **Nothing beats trading costs.** Four experiments: SPY, the 12 sector ETFs at two horizons, and 503
  S&P 500 stocks. Every model and rule loses money after a 1–2 bps round-trip cost, and no gross return has
  a 95% interval above zero.
- **Model complexity buys nothing.** No deep model (CNN1D, ResNet1D, InceptionTime, ResNet2D) is the best row
  in any experiment, and gradient-boosted trees are no clear improvement on logistic regression. Where one model
  edges past another, the margin is under half a basis point, far inside the intervals.
- **The entropy gate does not pick better days.** The default trend gate approves about as often as it would
  on pure noise. Of 93 approved-vs-other comparisons, 7 have intervals that exclude zero, about 5 would by
  chance, and 4 of the 7 point the wrong way.
- **Entropy adds nothing in any of three roles.** As a gate (above), as continuous model inputs (it moved
  returns by −0.45 to +0.27 bps with no consistent sign) and as inputs to the decision engine (which then
  traded two to five times as often and lost 0.1–0.6 bps more per opportunity).
- **The one lead was volatility, not skill.** Trading only a model's most confident predictions raises the
  return per trade, most for S&P 500 logistic regression (+6.15 bps on its top 10%). But those trades are 2–3
  times as volatile as average, and picking the most volatile moments without any model does as well or
  better, in bps and per unit of risk. Nothing beats costs through prediction: no model at any confidence
  level, not entropy in any role, not the RL engine.
- **A side finding, not yet checked.** Trading the models' mostly long direction at the S&P 500's most volatile
  moments earned +15.21 bps gross on the top 2% [+1.57, +29.18], in 10 of 14 quarters. It is a market-level
  effect, not stock prediction, and a flat 2 bps cost understates trading costs exactly then.
- **Permutation entropy has a blind spot that may explain part of the null.** Ordinal patterns of
  5-minute returns barely react to a slowly varying drift, the kind of trend multi-hour momentum relies
  on. In simulation they separate trending from noise spells by 0.07–0.17 standard deviations; patterns of
  hourly returns separate them by about 0.9, and a plain variance ratio by about 2.
- **With the market removed, models trained to call direction have no stock-level skill.** Scored where breadth
  counts, across stocks at each moment, every S&P 500 model's cross-sectional IC is about zero (−0.004 to +0.002), and the data could detect an
  IC half the size a profitable long-short book needs (0.0075 against 0.0154), so this null rules out a
  profitable signal of that kind. The 768 S&P 500 trades a day had amounted to only 7–14 independent bets.
- **One hint of stock-picking skill, which did not replicate and would not pay.** Trained to say which stocks beat
  their peers, the tree model ranked 85 Nasdaq-100 stocks it never saw better than chance (IC +0.0141, t = 3.37),
  but its 12-a-side book earns +1.47 bps gross per position, under a 2 bps cost: a null by the rule fixed in
  advance. The pre-registered replication then failed: on the 416 stocks it trained on, scored in later periods, the
  IC is +0.0005 (t = 0.2) and the timing IC is −0.0055. On the 85, the IC is mostly a fixed preference for some
  stocks, with timing skill indistinguishable from zero (+0.0019, t = 0.4).
- **The measurement works.** On synthetic data the gate passes planted trends and refuses zig-zags and tick
  noise, and the models recover planted signals. The null on real data is a finding, not a broken pipeline.
- **The decision engine (RL) finds nothing either.** On the ETFs it learns to stay flat. On the S&P 500 the
  first version collapsed into always-long, a flaw in the engine, not a market finding. The market-neutral
  version fixes that and also stays flat (−0.1 to −0.2 bps, 4–8% of samples traded): no stock-specific signal
  survives costs in the rules and logistic regression tested. Where it does learn, it learns the right thing:
  on synthetic data it finds planted edges and stays out of noise.

## Main case: selective trading (2026-10-04)

Trade only each model's strongest signals, across a wide universe so they still come often, with cutoffs from
earlier quarters only (`scripts/selective.py`, Colab section 13). On the S&P 500 the top 10% of logistic
regression's signals is about 65 trades a day; on the ETFs the top levels are 0.5–3.4 a day. It became the main
case because the one lead so far came from selectivity. The test: confident trades must earn more per unit of
their own risk (return ÷ trailing volatility × √hold, ×100, "risk-scaled") than a placebo that keeps the same
share of signals ranked by trailing volatility instead of confidence and trades the model's direction.

**Result: the lead is volatility, not skill.** S&P 500:

| Selection | Kept | Gross bps/trade | Hit rate | Risk-scaled [95% CI] | Volatility vs. average | Quarters net > 0 |
|---|---:|---:|---:|---|---:|---:|
| logreg, every signal | 100% | +0.89 | 0.503 | +0.0 [−1.3, +1.4] | 1.0× | 3/14 |
| logreg, top 10% by confidence | 8% | +6.15 | 0.512 | +1.7 [−1.3, +5.2] | 1.9× | 7/14 |
| top 10% by volatility (placebo) | 11% | +9.07 | 0.512 | +2.8 [−0.7, +7.9] | 2.6× | 8/14 |
| logreg, top 2% by confidence | 2% | +11.87 [+0.08, +24.40] | 0.524 | +2.4 [−1.3, +6.5] | 2.9× | 9/14 |
| top 2% by volatility (placebo) | 2% | +15.21 [+1.57, +29.18] | 0.525 | +2.9 [−0.0, +6.0] | 4.5× | 10/14 |
| gbm, top 2% by confidence | 3% | +4.76 | 0.537 | +4.9 [−0.0, +9.6] | 1.2× | 4/14 |
| top 2% by volatility (placebo) | 2% | +6.26 | 0.514 | +1.6 [−2.6, +5.3] | 4.5× | 7/14 |

- **Logistic regression picks volatile moments.** Its most confident trades are 1.9–2.9 times as volatile as
  average, and picking the most volatile moments directly does as well or better at every level, in bps and per
  unit of risk. Per unit of risk its confident trades earn +1.1 to +2.4 against +0.0 for all trades, every
  interval including zero.
- **The trees are a weak partial exception.** `gbm`'s and `gbm_ent`'s most confident S&P 500 trades are only
  1.2–1.5 times as volatile as average and beat the placebo per unit of risk (+3.0 to +4.9 against +1.6 to
  +3.5), with a hit rate of 0.53. But every interval touches zero, 95–98% of those trades are longs placed on
  fewer than half the days (a bet on which days the market rises, not on which stocks), and they made money in
  only 4–6 of 14 quarters.
- **ETFs: the placebo matches or beats confidence almost everywhere.** The one exception is logistic regression
  on the half-hour ETFs (top 10%: +5.2 [−2.6, +13.6] per unit of risk against −0.7 for the placebo), with
  intervals that wide and under one trade a day.

**A side finding, not yet checked.** Trading the models' direction, mostly long (68–95%), at the S&P 500's most
volatile moments earned large gross returns: +15.21 bps on the top 2% [+1.57, +29.18], positive in 10 of 14
quarters, and +8.81 to +9.50 on the top 10% with wide intervals. It is a market-level effect, not stock
prediction. One plausible reading is buying into selloffs and catching the rebounds, in line with the finding that
supplying liquidity pays more when volatility is high (Nagel 2012, "Evaporating Liquidity", Review of Financial
Studies). Before it counts for anything:

- **Costs:** a flat 2 bps understates trading costs exactly when spreads widen in volatile moments.
- **One episode?** It may be mostly the April 2025 selloff and rebound.
- **Market exposure:** it is unhedged; it has not been tested market-neutral.

## Diversification: we have fewer independent bets than trades (2026-10-04)

Every result above scores each trade against its own direction. Trades placed at the same moment share the
market's move, and the models are almost always long: 97% of `gbm`'s and 79% of logistic regression's S&P 500
signals, 95% and 84% of their top-10% trades. A day's hundreds of S&P 500 trades are therefore closer to one bet
on the market than to hundreds of bets, which is also why 673,694 S&P 500 trades gave intervals no narrower than
14,005 ETF trades, why the engine collapsed into always-long, and why the volatile-moment side finding is
68–95% long. The 12 ETFs are one asset (SPY is the sum of the 11 sector ETFs), and one quarter (April 2025)
carries several results.

**Built** (`scripts/breadth.py`, with synthetic controls in `tests/test_crosssection.py`): effective independent
bets per day, the cross-sectional IC (market-neutral, optionally sector-neutral), market- and sector-neutral
long-short books, how alike the models are, and the IC the data could detect against the IC a long-short book
needs to pay its cost. Plus `configs/cross_asset_multihour.yaml`: 37 ETFs in 7 asset classes, for breadth from
different drivers.

### Results (2026-10-05)

**How many independent bets a day's trading is**, S&P 500 (positions per day · net long exposure · effective
independent bets per day):

| model | every signal | its top 10% | long 10% / short 10% |
|---|---|---|---|
| `gbm` | 768 · +0.93 · **7.0** | 57 · +0.90 · **2.4** | 154 · 0.00 · **75.6** |
| `gbm_ent` | 768 · +0.93 · **7.1** | 35 · +0.94 · **5.8** | 154 · 0.00 · **67.1** |
| logistic regression | 768 · +0.59 · **14.4** | 65 · +0.55 · **12.4** | 154 · 0.00 · **48.5** |
| momentum_window rule | 762 · +0.02 · **67.7** | 758 · +0.03 · **70.3** | 154 · 0.00 · **100.1** |

The selective main case was the least diversified of all (2.4 bets a day for the trees' top 10%), which is why its
results were so lumpy. A long-short book turns the same predictions into 50–100 independent bets a day.

**Skill across stocks with the market removed**, S&P 500 (rank IC between score and return across stocks at each
moment, averaged over days; long-short book of the top and bottom 10%, net of 2 bps per position):

| model | IC [95% CI] | long-short net | sector-neutral IC [95% CI] |
|---|---|---:|---|
| `gbm` | −0.0037 [−0.0079, +0.0008] | −2.08 | −0.0040 [−0.0076, −0.0002] |
| `gbm_ent` | −0.0032 [−0.0081, +0.0017] | −2.38 | −0.0029 [−0.0067, +0.0011] |
| logistic regression | +0.0014 [−0.0039, +0.0070] | −0.86 | +0.0019 [−0.0026, +0.0063] |
| momentum_day rule | +0.0021 [−0.0056, +0.0090] | −1.68 | +0.0022 [−0.0034, +0.0079] |
| average of the three trained models | −0.0018 [−0.0069, +0.0034] | −1.54 | −0.0017 [−0.0058, +0.0024] |

- **A profitable signal of this kind is ruled out.** Stocks differ from each other by about 74 bps over the hold,
  so a 10%-a-side book needs an IC of at least 0.0154 to pay 2 bps; the data would detect an IC of 0.0075 (2.8
  standard errors), and every model's interval stays below 0.009. At a 1 bp cost the bar halves to about 0.008,
  at the edge of what the data can see, so for very cheap execution the null is less decisive.
- **The models are not diverse either.** Their scores correlate 0.26 on average: three models are about 2.0
  independent ones (on the ETFs, eight models are about 3.4).

**The other universes.** On the 12 ETFs stocks differ by only 13–15 bps, so a book needs an IC of about 0.08,
far above anything measured (all within ±0.03). A few ETF intervals exclude zero (logistic regression −0.028 and
InceptionTime −0.024 on the 4-hour ETFs; `gbm_ent` +0.024 and the ensemble +0.025 on the half-hour ETFs), about
what chance gives across some 25 comparisons. On the cross-asset universe (33 of the 37 ETFs had data in the run)
every IC is within ±0.012, but there the data can only detect an IC of 0.034 against 0.029 needed, so that null
does not rule a signal out; its sweep is the same null as everywhere (gross +0.05 to +0.79 bps, net negative for
every model).

## Hold-out test: trade stocks the models never saw (protocol fixed 2026-10-07)

Train on the S&P 500 without the Nasdaq-100's members, trade the Nasdaq-100 members, rank them and hold a
diversified long-short book (`configs/ndx_holdout.yaml`, Colab section 16).

**Universes** (lists from Wikipedia: S&P 500 on 2026-09-29, Nasdaq-100 on 2026-10-07): 85 of the Nasdaq-100's
100 members are in the S&P 500 list we trained on, so they are held out of training; the other 418 S&P 500
stocks train the models; the 15 members outside the S&P 500 (ALAB, ALNY, ARM, ASML, CCEP, CRWV, FER, MELI, MSTR,
NBIS, PDD, RKLB, SHOP, SPCX, TRI) are left out, with no comparable history. The 85 are concentrated: 35 are
Information Technology.

**Protocol, fixed before any result.** Target: did the stock beat its universe's average at that moment. Models:
logistic regression, `gbm`, `gbm_ent`, and their average (4 looks). Book: 12 long and 12 short of the 85 at each
moment, 2 bps per position. A finding needs both, for the same model: a cross-sectional IC with t ≥ 3, and a
12-a-side book whose net 95% interval is above zero. Anything else is a null, reported with the IC the data can
detect and the IC a book needs.

**What to expect from this much data** (estimates before the run, to compare with what the report computes):
with about 85 stocks the smallest detectable IC is roughly 0.014–0.018, against 0.0075 for the full S&P 500
(scaled from the runs we have with 12, 33 and 501 symbols); a 12-a-side book needs an IC of roughly 0.012–0.017 to
pay 2 bps if stocks differ by 74–95 bps. So a null here may not rule out a signal that would just pay. Its value is
the other direction: a positive result on stocks the models never saw, against a rule fixed in advance, would be
the most credible one this project could produce.

**Caveats:** a hold-out across stocks, not across time (same dates and market episodes); today's lists, and the
Nasdaq-100's in particular, are survivorship-biased toward stocks that rose.

### Result (2026-10-07): a null by the protocol, with the project's first significant ranking skill

Trained on 418 stocks, scored on the 85 they never saw, 940 test days, 12 long and 12 short at each moment:

| model | IC [95% CI] | t | book gross [95% CI] | net after 2 bps |
|---|---|---:|---|---:|
| `gbm` | **+0.0141** [+0.0060, +0.0221] | **3.37** | **+1.47** [+0.45, +2.51] | −0.53 |
| `gbm_ent` | +0.0141 [+0.0051, +0.0231] | 3.04 | +0.93 [−0.24, +2.09] | −1.07 |
| average of the three trained models | +0.0138 [+0.0050, +0.0224] | 3.09 | +0.98 [−0.14, +2.04] | −1.02 |
| logistic regression | +0.0057 [−0.0028, +0.0140] | 1.38 | +0.34 [−0.97, +1.50] | −1.66 |
| momentum rules | −0.0036 to +0.0010 | < 1 | +0.05 to +0.78 | below −1.2 |

- **By the rule, a null.** `gbm` clears the skill half (t = 3.37 ≥ 3), but its book's net interval after 2 bps,
  about [−1.55, +0.51], is not above zero, and the rule needs both.
- **Still the first evidence of ranking skill on data a model never saw.** On the S&P 500, the same tree model
  trained to call each stock's direction scored an IC of −0.0037: asking which stocks beat their peers is what
  changed. Both sides of the book earn (+1.26 bps on the longs, +1.68 on the shorts), and about 70% of the IC holds
  within sectors (sector-neutral IC +0.0099, t = 2.56), so it is not one sector bet.
- **Why it does not pay.** Because of IEX gaps, only about 56 of the 85 stocks have a complete window at a typical
  moment, so 12 a side is the top and bottom fifth, and the book needs an IC of 0.0204 to pay 2 bps; `gbm` has
  0.0141. Its gross 1.47 bps per position would pay only if a round trip cost under about 1.5 bps.
- **Reasons for caution:** both lists are today's members, and a fixed preference for stocks that kept rising would
  look like skill (survivorship); `gbm_ent` (3.04) and the average (3.09) clear the bar only just, after many tests
  in this project; and this is one hold-out sample, across stocks but not across time.

### Checks and replication (2026-10-07): the skill did not replicate

**Rule, fixed before running** (committed in `079f520` at 17:55 on 2026-10-07; the configs and the earlier
version of this section say 2026-10-08, a mislabelled date, and the commit time is the one that counts): `gbm`
replicates only if its IC on the 418 training stocks has t ≥ 3 **and** its timing IC's 95% interval is above zero.
The 418 are scored in later periods only (out of time, not out of universe); the 85 are scored in the same run.
The report scored 416 of the 418 (the two others are not in its predictions).

| `gbm` | stocks | IC [95% CI] · t | net of stock averages | static | timing |
|---|---:|---|---|---|---|
| hold-out | 85 | +0.0146 [+0.0057, +0.0233] · 3.3 | +0.0103 [+0.0014, +0.0190] · 2.3 | +0.0137 [−0.0024, +0.0286] · 1.7 | +0.0019 [−0.0074, +0.0119] · 0.4 |
| **replication** | 416 | **+0.0005** [−0.0048, +0.0057] · **0.2** | −0.0019 [−0.0072, +0.0033] · −0.7 | +0.0066 [−0.0034, +0.0158] · 1.4 | **−0.0055** [−0.0117, +0.0006] · −1.8 |

(The 85 use at least 20 stocks per moment, as the skill report does, hence +0.0146 here against +0.0141 in the
breadth report below.)

- **Verdict: does not replicate.** The IC on the 416 is zero (t = 0.2, positive in 8 of 15 quarters, sign test
  p = 0.50) and the timing IC is slightly negative, so both halves of the rule fail.
- **The 85 reproduce exactly**: IC +0.0141, t = 3.4, book gross +1.47 bps and net −0.53, 23.8 independent bets a
  day, identical to the hold-out run. The earlier result was not a bug. It did not generalize.
- **On the 416, a profitable signal is ruled out.** The long-short book earns −0.15 bps gross [−1.06, +0.76]
  (−2.15 net), the sector-neutral IC is −0.0006, and the data could detect an IC of 0.0073 against the 0.0138 a
  book needs to pay.
- **On the 85 the IC is mostly a fixed tilt, not timing.** Timing is about zero (+0.0019, t = 0.4). The static part
  is positive but not significant on its own (+0.0137, t = 1.7), and removing each stock's average return shrinks the
  IC to +0.0103 (t = 2.3), under the t ≥ 3 bar. A fixed preference for stocks that kept rising is what survivorship
  in today's Nasdaq-100 list can reward. This is an explanation consistent with the numbers, not a tested one.
- **Spread over time, but not independent evidence.** On the 85, `gbm` and `gbm_ent` have a positive IC in 12 of 15
  quarters (sign test p = 0.018; `logreg` 10 of 15), the largest quarter holds 13–19% of the positive total, and
  all three signal times (10:25, 11:25, 12:25) are positive (+0.0125 to +0.0148, t 2.0–2.3). It is the same 85
  stocks and the same dates as the original IC, so it supports that number without adding a second sample.
- **A model with general stock-picking skill would rank the 416 too.** It ranks them at chance, and ranks the 85
  better. The simplest reading is that the 85 share a property (large technology and growth names) that happened to
  line up with what the model scores in 2022–2026.
- **It would not pay either way.** A book on the 85 needs an IC of 0.0205 to cover 2 bps, and the observed IC is
  0.0141–0.0146. Its gross +1.47 bps would pay only if a round trip cost under about 1.5 bps.

**What goes in the thesis:** a relative-target tree model ranked unseen Nasdaq-100 stocks above chance (t = 3.4, 12 of
15 quarters positive), but the effect did not replicate on the other stocks, was not timing skill, and was too small
to cover costs. Every step, including the replication rule, was fixed before the data was read.

## Common setup

| | |
|---|---|
| Data | Alpaca's free IEX feed: 5-minute bars, regular hours, from 2020-07-27 (where the free history begins). IEX carries only part of the consolidated tape. |
| Validation | 15 quarterly walk-forward folds: 504 training days, 63 validation, 63 test. 945 test days, from late 2022 to August 2026. |
| Scoring | Gross basis points per trade; net = gross − round-trip cost. 95% intervals resample whole days, because trades on the same day move together. Learned models are averaged over 3 training seeds. |
| Gate | Trend statistic against 99 shuffled copies, α = 0.05, on the last 5 sessions (390 bars) with patterns of length m = 4. A reading at one session's close applies to the next session. |

| Config | Universe | Signal → hold | Cost | Status |
|---|---|---|---:|---|
| `spy_gao` | SPY | first half-hour → last half-hour | 1 bps | run |
| `etf_gao` | 12 ETFs | first half-hour → last half-hour | 2 bps | run |
| `etf_intraday` | 12 ETFs | 4-hour window → 1-hour hold | 2 bps | run |
| `sp500_multihour` | 503 stocks | 1-hour window → 2.5-hour hold | 2 bps | run: rules, logreg and trees |
| `etf_multihour` | 12 ETFs | 1-hour window → 2.5-hour hold | 2 bps | not run yet |

## Does anything beat costs?

Gross basis points per trade with 95% intervals, and net at each experiment's cost. For the tree models,
params counts the leaves of the first quarter's model; `_ent` models also see the entropy features.

**`spy_gao`** (net at 1 bps)

| model | params | accuracy [95% CI] | gross bps/trade [95% CI] | net |
|---|---:|---|---|---:|
| always_up | 0 | 0.507 [0.474, 0.540] | +0.29 [−1.21, +1.63] | −0.71 |
| momentum_day | 0 | 0.508 [0.476, 0.541] | +0.56 [−0.78, +1.95] | −0.44 |
| logreg | 50 | 0.503 [0.470, 0.533] | −0.29 [−1.81, +1.12] | −1.29 |
| resnet1d | 505,730 | 0.490 [0.458, 0.521] | −0.13 [−1.54, +1.29] | −1.13 |
| gbm | 15 | 0.470 [0.437, 0.502] | −0.50 [−1.95, +0.90] | −1.50 |
| gbm_ent | 15 | 0.477 [0.443, 0.510] | −0.23 [−1.66, +1.16] | −1.23 |
| resnet1d_ent | 506,014 | 0.476 [0.442, 0.509] | −0.58 [−2.10, +0.84] | −1.58 |

**`etf_gao`** (net at 2 bps)

| model | params | accuracy [95% CI] | gross bps/trade [95% CI] | net |
|---|---:|---|---|---:|
| always_up | 0 | 0.494 [0.473, 0.516] | −0.54 [−1.88, +0.76] | −2.54 |
| momentum_day | 0 | 0.517 [0.502, 0.531] | +0.52 [−0.33, +1.40] | −1.48 |
| logreg | 50 | 0.502 [0.489, 0.516] | +0.58 [−0.21, +1.47] | −1.42 |
| resnet1d | 505,730 | 0.505 [0.493, 0.519] | +0.20 [−0.50, +0.92] | −1.80 |
| gbm | 90 | 0.503 [0.486, 0.519] | +0.45 [−0.48, +1.44] | −1.55 |
| gbm_ent | 135 | 0.510 [0.493, 0.525] | +0.50 [−0.43, +1.42] | −1.50 |
| resnet1d_ent | 506,014 | 0.505 [0.492, 0.517] | +0.17 [−0.62, +1.06] | −1.83 |

**`etf_intraday`** (net at 2 bps)

| model | params | accuracy [95% CI] | gross bps/trade [95% CI] | net |
|---|---:|---|---|---:|
| always_up | 0 | 0.514 [0.498, 0.530] | +0.70 [−0.42, +1.81] | −1.30 |
| momentum_day | 0 | 0.509 [0.498, 0.519] | −0.06 [−0.98, +0.78] | −2.06 |
| momentum_window | 0 | 0.505 [0.494, 0.517] | −0.17 [−1.21, +0.77] | −2.17 |
| logreg | 386 | 0.503 [0.492, 0.515] | +0.32 [−0.53, +1.24] | −1.68 |
| cnn1d | 175,042 | 0.509 [0.496, 0.522] | +0.11 [−0.71, +0.96] | −1.89 |
| resnet1d | 505,730 | 0.513 [0.499, 0.526] | +0.40 [−0.55, +1.31] | −1.60 |
| inceptiontime | 472,066 | 0.512 [0.500, 0.524] | −0.00 [−0.79, +0.85] | −2.00 |
| resnet2d (GAF images) | 914,690 | 0.509 [0.497, 0.522] | +0.32 [−0.50, +1.10] | −1.68 |
| gbm | 150 | 0.512 [0.498, 0.527] | +0.71 [−0.28, +1.72] | −1.29 |
| gbm_ent | 30 | 0.510 [0.495, 0.526] | +0.47 [−0.57, +1.47] | −1.53 |
| resnet1d_ent | 506,014 | 0.511 [0.498, 0.525] | +0.48 [−0.43, +1.34] | −1.52 |

**`sp500_multihour`** (net at 2 bps, 673,694 trades)

| model | params | accuracy [95% CI] | gross bps/trade [95% CI] | net |
|---|---:|---|---|---:|
| always_up | 0 | 0.506 [0.495, 0.516] | +1.03 [−1.61, +3.84] | −0.97 |
| momentum_day | 0 | 0.496 [0.490, 0.501] | −0.87 [−2.28, +0.40] | −2.87 |
| momentum_window | 0 | 0.496 [0.492, 0.500] | −0.63 [−1.51, +0.23] | −2.63 |
| logreg | 98 | 0.503 [0.496, 0.510] | +0.86 [−0.95, +3.01] | −1.14 |
| gbm | 15 | 0.504 [0.494, 0.515] | +0.71 [−1.91, +3.44] | −1.29 |
| gbm_ent | 15 | 0.505 [0.495, 0.515] | +0.85 [−1.71, +3.63] | −1.15 |

What the tables say:

- **The best row in each experiment is a rule, logistic regression or the trees.** On the 4-hour ETFs, `gbm`
  (+0.71) edges always_up (+0.70) by a hundredth of a basis point. No deep model is best anywhere.
- **The trees found little to learn.** On SPY and the S&P 500 the validation fold stopped them after a single
  tree (15 leaves) in the first quarter; on the ETFs after 2 to 10 trees.
- **Image view (Q4):** ResNet2D on Gramian Angular Field images, with 914,690 parameters, earns +0.32 against
  ResNet1D's +0.40 on the same windows. The image encoding adds nothing.
- **The one accuracy interval clear of 50%** is half-hour momentum across the ETFs: 0.517 [0.502, 0.531]. The
  last half-hour moves in the direction of the return from the previous close to 10:00 slightly more often
  than not, the direction of the published intraday momentum effect (Gao, Han, Li and Zhou 2018). It is worth
  +0.52 bps a trade, about a quarter of the cost.
- **Single stocks lean the other way.** On the S&P 500 both momentum rules are right 49.6% of the time, a hint
  of short-term reversal. Fading them would earn under 1 bps gross, still below cost.
- **At the longer horizons, simply holding long beat or tied every model,** and that is mostly market drift. On the
  S&P 500 one quarter (February–May 2025, which includes the April 2025 selloff and rebound) earned +10.8 bps per
  trade for always_up, ten times its average.
- **The intervals stay wide even with 673,694 S&P 500 trades.** Stocks move together within a day, so the
  effective sample is closer to 945 days than to 673,694 trades. Going from 12 ETFs to 503 stocks bought much
  less statistical power than the trade count suggests.

## Does the gate pick better days? (Q1)

Share of test samples falling on days the gate approved:

| Experiment | trend (default) | entropy | weighted entropy |
|---|---:|---:|---:|
| `spy_gao` | 4.9% | 6.8% | 13.8% |
| `etf_gao` | 3.2% | 6.5% | 9.2% |
| `etf_intraday` | 3.4% | 6.7% | 10.0% |
| `sp500_multihour` | 3.6% | 5.9% | 16.0% |
| pure noise (simulation) | about 5% | about 5% | about 5% |

- **Trend** approves at or slightly below the noise rate. Readings on tick-limited or stale prices are refused
  outright, which pulls it down.
- **Weighted entropy** approves two to three times as often as noise would. It reads volatility clustering as
  order, and each 5-session window overlaps the next four, so one volatile episode is approved on several
  consecutive days. In simulation, one planted jump was approved on 9 separate days.

The same predictions, split by whether the gate approved the day: 93 comparisons (model × statistic ×
experiment). Seven have 95% intervals that exclude zero, where chance alone would give about 5, and four of
the seven point the wrong way:

| Experiment | Model | Gate | Approved − other, bps [95% CI] |
|---|---|---|---|
| `spy_gao` | logreg | trend | −9.57 [−17.47, −2.85] |
| `spy_gao` | logreg | weighted entropy | +4.48 [+0.66, +8.34] |
| `etf_gao` | logreg | entropy | +1.97 [+0.04, +3.99] |
| `etf_intraday` | always_up | weighted entropy | −1.47 [−3.07, −0.03] |
| `etf_intraday` | inceptiontime | entropy | +1.67 [+0.01, +3.29] |
| `spy_gao` | gbm | entropy | −6.62 [−12.83, −1.44] |
| `etf_intraday` | gbm_ent | weighted entropy | −2.17 [−3.71, −0.68] |

- **The default trend gate:** 1 of its 31 comparisons excludes zero, and that one is negative.
- **Within each symbol:** the gate refuses whole low-priced symbols, so the raw split partly compares one
  symbol with another. After subtracting each symbol's own average, 3 of 72 comparisons exclude zero, about what
  chance gives (3.6).
- **The best-looking ETF lead did not replicate.** At 4h→1h on the ETFs, trend-approved days looked better for
  momentum_window (+2.47 [−0.19, +5.00]) and resnet1d (+2.02 [−0.42, +4.61]). On the S&P 500, momentum_window
  shows +0.22 [−1.17, +1.47].

## Trading only confident predictions

Each fold's confidence cutoff is set from earlier folds only. Gross bps per trade at each share of trades kept:

| Experiment, model | 100% | 50% | 20% | 10% [95% CI] | net at 10% |
|---|---:|---:|---:|---|---:|
| `sp500_multihour`, logreg | +0.89 | +1.90 | +3.96 | +6.15 [−1.56, +14.80] | +4.15 |
| `sp500_multihour`, gbm | +0.95 | +2.83 | +2.21 | +3.91 [−2.39, +10.05] | +1.91 |
| `sp500_multihour`, gbm_ent | +1.08 | +2.09 | +0.84 | +3.15 [−3.73, +9.91] | +1.15 |
| `etf_intraday`, resnet1d | +0.47 | +1.06 | +1.70 | +2.36 [−0.27, +4.84] | +0.36 |
| `etf_intraday`, gbm | +0.48 | +0.45 | +0.99 | +1.97 [+0.00, +4.09] | −0.03 |
| `etf_intraday`, resnet1d_ent | +0.40 | +0.85 | +0.69 | +1.59 [−0.93, +3.91] | −0.41 |
| `etf_intraday`, gbm_ent | +0.29 | +0.50 | +1.06 | +1.26 [−0.67, +3.29] | −0.74 |
| `etf_intraday`, cnn1d | +0.09 | +0.18 | +0.23 | +0.58 [−1.96, +2.79] | −1.42 |
| `etf_intraday`, resnet2d | +0.39 | +0.32 | +0.42 | +0.34 [−1.24, +1.90] | −1.66 |
| `etf_intraday`, logreg | −0.18 | −0.07 | −0.02 | −0.53 [−4.47, +3.90] | −2.53 |
| `etf_intraday`, inceptiontime | −0.11 | −0.39 | −0.40 | −1.22 [−3.95, +1.23] | −3.22 |
| `etf_gao`, logreg | +0.66 | +0.75 | +1.15 | +1.74 [−2.09, +5.88] | −0.26 |
| `etf_gao`, resnet1d | +0.27 | +0.57 | +0.47 | −0.02 [−3.92, +4.23] | −2.02 |
| `etf_gao`, resnet1d_ent | +0.31 | +0.52 | −0.02 | −0.17 [−5.03, +5.00] | −2.17 |
| `etf_gao`, gbm_ent | +0.68 | +0.64 | +0.42 | −0.80 [−6.04, +3.88] | −2.80 |
| `etf_gao`, gbm | +0.61 | −0.42 | −0.91 | −2.32 [−6.63, +1.24] | −4.32 |
| `spy_gao`, gbm | −0.04 | −1.70 | +1.00 | +1.08 [−2.68, +4.98] | +0.08 (at 1 bps) |
| `spy_gao`, resnet1d_ent | −0.72 | +0.70 | +1.26 | +0.03 [−5.50, +5.34] | −0.97 (at 1 bps) |
| `spy_gao`, resnet1d | −0.61 | −0.72 | −1.69 | −0.60 [−6.67, +5.21] | −1.60 (at 1 bps) |
| `spy_gao`, gbm_ent | +0.23 | −1.97 | −2.10 | −2.11 [−7.66, +3.24] | −3.11 (at 1 bps) |
| `spy_gao`, logreg | +0.33 | −1.26 | +0.44 | −2.17 [−10.39, +5.76] | −3.17 (at 1 bps) |

The 100% column differs slightly from the main tables because early folds without enough history to set a
cutoff are dropped at every level. The realized share kept also differs from the target, a lot for the
trees, whose probabilities come in steps: the S&P 500 logreg 10% level keeps 8%, the S&P 500 trees keep 3%,
and `gbm` on the 4-hour ETFs keeps 22%.

Return rises with confidence for all three S&P 500 models and for five of eight on the 4-hour ETFs, and
rarely on the half-hour setups. It is not an edge yet:

- **Every interval includes zero.**
- **The levels are nested:** the 10% trades are part of the 20% trades, so each row is one result, not four.
- **Accuracy barely moves.** S&P 500 logreg goes from 0.503 to 0.512 and the trees to 0.530 at most, so the gain
  comes from bigger moves, not more correct calls. That fits models that are most confident on volatile days,
  when every move is larger: logistic models and trees give their most extreme probabilities when their inputs
  are most extreme.
- **The models are not independent witnesses.** They share inputs and training data, so several of them showing
  the slope is not several confirmations.
- **The check found volatility.** Scored per unit of their own volatility, against a placebo that picks the most
  volatile moments instead, the confident trades do no better (Colab section 13; see "Main case" above).

## Checks on the measurement itself

**The pipeline finds structure when it is there** (`scripts/smoke_test.py`, synthetic data, as run on Colab):

| Planted structure | Result |
|---|---|
| Pure noise | trend gate approves 4.7% (plain entropy 5.8%) |
| Trending series | trend gate approves 45.1% |
| Zig-zag, like bid-ask bounce | trend gate 0.0%; plain entropy 80.3%, which is why it is not the default |
| Low-priced, tick-limited series | 100% refused |
| Persistent intraday drift | logreg accuracy 0.649, resnet1d 0.667 |
| Half-hour momentum | momentum_day accuracy 0.825, logreg 0.742 |

**Permutation entropy of real returns reads like noise.** With m = 4 there are K = 24 patterns and N = 387 per
reading. Finite samples bias the entropy down by about (K − 1)/(2N) nats, so pure noise reads about 0.991
instead of 1. SPY reads 0.991. That is why the gate compares each reading with shuffled copies of the same
returns instead of a fixed threshold.

**Weighted permutation entropy.** Raw weighted entropy is biased by fat tails: in simulation its effective
sample shrinks from 390 bars to roughly 90–250. Tested against shuffled copies, its false-pass rate stays near
5%, for Student-t tails from 3 to 30 degrees of freedom and GARCH persistence from 0 to 0.95. Its high approval
rate on real data comes from overlapping windows instead, so approvals should be counted as episodes, not days.
That count is not built yet.

**Problems found and fixed before these results** (details in the README's design notes):

1. A gate reading taken at 4 pm could approve an 11 am trade on the same day. Readings now apply to the next
   session.
2. A per-window z-score set every window's cumulative return to zero, hiding the trend from the models. On
   planted drift that cost logistic regression 14 points of accuracy (0.651 against 0.793 with training-fold
   standardization).
3. Plain entropy passed zig-zags from bid-ask bounce, and weighted entropy passed fat tails. The shuffle test,
   the trend statistic and the tick-limited and stale refusals fixed both.
4. Two identical reruns of one model differed about as much as different architectures did. Training is now
   deterministic, with 3 seeds per model.
5. Moves under 5 bps were dropped from the test set as well as from training. Now only training drops them.

## Decision engine (reinforcement learning)

Built in `src/engine.py` and `scripts/engine.py`. For each sample a policy chooses short, flat or long from the
models' confidences and the gate's verdict, and is paid position × return − cost. It learns online, fold by
fold, from earlier feedback only (REINFORCE, a contextual bandit). It is compared with the best fixed rule
chosen from the past, and it learns from one of two kinds of feedback:

- **bandit:** the reward of the action it took.
- **full:** all three actions scored on every past sample, including the trades it skipped.

Synthetic controls (`tests/test_engine.py`), in net bps per opportunity after a 2 bps cost:

| Control | Bandit feedback | Full feedback |
|---|---|---|
| Clear edge | +1.57 | +1.55 |
| Edge barely above cost, two data seeds | 0.00 and +0.50 | +0.38 and +0.51 |
| Pure noise, two data seeds | 0.00 and 0.00 | 0.00 and 0.00 |
| Edge only on gate-approved samples, gate given as input, three data seeds | +0.32, 0.00, +0.35 | +0.59, +0.60, +0.59 |
| Same, gate not given | 0.00 on all three | −0.02, +0.02, +0.01 |

- **Full feedback learns small edges reliably.** Bandit feedback learned the barely-above-cost edge on one seed
  and never traded on the other.
- **The gate matters only when the edge lives on approved days.** In the last two rows, full feedback found the
  edge only when the gate's verdict was an input.
- **Copying hindsight is the wrong way to learn from missed trades.** A policy trained to copy each sample's
  best action in hindsight traded every sample and lost 1.98 and 1.91 bps per opportunity on pure noise. Missed
  opportunities are therefore measured by situation (gate verdict × signal strength) in `engine.md`.

### Real data, first run (2026-10-01)

Net bps per opportunity after a 2 bps cost, folds 1–14, engines averaged over 3 training seeds; share of
samples traded in brackets. Staying flat is 0 by construction.

| | `etf_intraday` | `etf_gao` | `sp500_multihour` |
|---|---:|---:|---:|
| always long | −1.54 | −2.73 | −0.94 |
| best fixed rule, chosen from the past | −0.50 [−0.84, −0.17] (23%) | −0.14 [−0.73, +0.38] (14%) | +0.18 [−0.07, +0.46] (2%) |
| engine, bandit, models only | −0.08 [−0.34, +0.17] (7%) | −0.04 [−0.35, +0.30] (8%) | **−2.98** [−4.97, −1.21] (76%) |
| engine, bandit, with gate | −0.15 [−0.40, +0.09] (13%) | +0.00 [−0.32, +0.35] (7%) | −1.17 [−1.89, −0.49] (23%) |
| engine, full feedback, models only | −0.19 [−0.49, +0.10] (11%) | −0.27 [−0.65, +0.14] (11%) | −1.52 [−4.02, +1.34] (98%) |
| engine, full feedback, with gate | −0.18 [−0.45, +0.08] (14%) | −0.17 [−0.50, +0.20] (9%) | −1.64 [−3.32, +0.32] (74%) |

- **ETFs: the engine learned to stay out.** Every ETF interval includes zero. It beats trading every signal
  (−1.5 to −2.2 bps) and, on `etf_intraday`, the rule chosen from the past. From fold 7 (`etf_intraday`) or
  fold 4 (`etf_gao`) it sits at exactly 0 almost every fold. An agent that learns from feedback finds no edge
  either, which matches the sweeps.
- **Neither addition helped on real data.** The gate approves 3–4% of samples, 50–100 per cell, too few for
  the engine to learn from, so the gate's effect on the engine is not testable at this size. Full feedback
  traded more and earned slightly less than bandit on the ETFs.
- **Missed opportunities: none found.** No cell has an interval above zero in any experiment. Choosing each
  group's best action in hindsight, which flatters it, earns only +0.05, +0.05 and +0.28 bps per opportunity.
- **S&P 500: the engine failed, because of how it was built.** It traded 76–98% of samples and lost. Under
  full feedback its P&L equals always-long's net to the hundredth in 8 of 9 folds from fold 6 on (for example
  +8.82 in fold 9 and −3.79 in fold 8). The market's direction flips from quarter to quarter, every stock shares
  it, and the engine treated one quarter's drift as tens of thousands of independent confirmations of "go
  long". The best row (+0.18) is one fold: the April 2025 rebound supplies about 1.90 of its 2.28 total.
- **Learning from all past quarters instead of the latest does not fix it** (synthetic: −1.73 to −1.94 against
  −1.85 to −2.33), because there are only about 14 independent quarters. Measuring returns relative to the other
  stocks at each date and bar does: in the same synthetic test the engine stays at 0.00 with no stock-specific
  edge and earns about +1.1 bps every time when 6 bps is planted, where the raw engine loses on noise and finds
  the edge only erratically (from −0.9 to +1.3 bps across seeds).
  That mode is `scripts/engine.py --neutral`; its S&P 500 result follows.

### Real data, market-neutral S&P 500 (2026-10-02)

The same 503 stocks and saved predictions (rules and logistic regression), with every return and model
confidence measured relative to the other stocks at the same date and bar. Net bps per opportunity after
2 bps charged on the stock leg only, folds 1–14:

| policy | net [95% CI] | traded |
|---|---|---:|
| always long (stock vs. the market) | −2.00 | 100% |
| trade every logreg signal | −1.43 [−2.05, −0.77] | 100% |
| trade every momentum_day signal | −1.29 [−1.96, −0.56] | 100% |
| trade every momentum_window signal | −1.45 [−1.98, −0.88] | 100% |
| best fixed rule, chosen from the past | +0.07 [−0.03, +0.18] | 1% |
| engine, bandit, models only | −0.11 [−0.20, −0.02] | 7% |
| engine, bandit, with gate | −0.18 [−0.28, −0.09] | 8% |
| engine, full feedback, models only | −0.16 [−0.26, −0.06] | 7% |
| engine, full feedback, with gate | −0.17 [−0.25, −0.08] | 4% |

- **The collapse is fixed.** The engine trades 4–8% of samples (it was 76–98%) and loses 0.1–0.2 bps (it was
  1.5–3.0). The small remaining loss is mostly the first learning step: fold 1 supplies about 70% of it, and
  from fold 4 on the engine sits at about 0.
- **No stock-specific signal survives cost.** Trading every signal relative to the market has a gross return
  of +0.57 (logreg), +0.71 (momentum_day) and +0.55 (momentum_window) bps, about a third of the cost. The lower
  ends of those intervals are near zero (−0.05, +0.04, +0.02), and the three signals are highly correlated, so
  they are not three confirmations. The breakeven cost of about 0.6 bps is below any realistic cost, before
  paying to hedge the market leg.
- **No missed opportunities.** In all ten groups (gate verdict × signal strength) the best action is to stay
  flat, so choosing in hindsight earns +0.00 bps per opportunity.
- **The gate does not help the engine:** −0.18 and −0.17 with it, −0.11 and −0.16 without. About 23,500
  samples (3.5%) are gate-approved, roughly 4,700 per cell.
- **What this does not rule out.** Only rules and logistic regression ran at this scale. The logistic model
  was trained to predict each stock's raw direction, not to rank stocks against each other, and the deep
  models were never run on the S&P 500. This is a null on those inputs, not on stock-specific prediction.

## Entropy as features instead of a gate (2026-10-03)

The gate was replaced by continuous entropy features that the models and the engine weigh for
themselves (design in the README, "Entropy as features"). Three models were added in pairs that differ
only by those features: gradient-boosted trees `gbm` / `gbm_ent`, and `resnet1d` / `resnet1d_ent` with
the same deep encoder. The engine now reads the continuous features instead of the gate's flags.

**What permutation entropy can and cannot see.** Ordinal patterns of returns depend on how each return
ranks against its neighbours. A drift that barely changes over three bars lifts all three alike, so the
ranks, and the patterns, look like noise. For a Gaussian process the monotone share at m = 3 depends on
the autocorrelations only through (2ρ1 − 1 − ρ2) / (2(1 − ρ1)): a slow drift (ρ1 = 0.35, ρ2 ≈ 0.34) reads
0.335 against 0.333 for noise, while a short-memory process with the same ρ1 = 0.35 reads 0.395.
Simulated markets switching between trending spells and noise spells (equal volatility):

| statistic, trailing 5 sessions | separation of trending from noise spells |
|---|---:|
| monotone share, 5-minute returns | 0.07–0.17 sd |
| monotone share, hourly returns | 0.86–0.88 sd |
| variance ratio (not an entropy measure) | 1.85–2.22 sd |

Hourly-scale entropy features were added for this reason. Whether the 5-minute gate's null on real data
comes from this blind spot is a hypothesis, not yet a result.

**The synthetic regime control did not favour entropy, and that is informative.** In the same simulated
market the trees without entropy already placed 82% of their most confident 30% of trades in trending
spells, and 97% of the top 10%: a strong drift shows up as a large window move, so the window itself
says when to trust the signal. With entropy the trees did no better (top-30% accuracy 0.764 against
0.792). Entropy can only add something the price window does not already show.

**A design problem found and fixed in the deep model.** With the window mostly noise, the 500k-parameter
body memorized it within an epoch or two, and early stopping froze a plain concatenation before the entropy
inputs were used: 0.50 accuracy where 0.75 was available, in a test where only entropy carried the signal.
Fitting the entropy path first as a logistic regression fixed it (0.745–0.771), and with uninformative
entropy it stays at chance (0.48–0.50).

**Controls** (`tests/test_entropy_features.py`, `tests/test_ml.py`, `tests/test_engine.py`): the features
match a direct calculation, never read past the signal bar and are missing rather than invented when
history is short; `gbm_ent` uses a signal that only entropy reveals and `gbm` cannot; the engine earns
+0.15 to +0.43 bps per opportunity on an edge that only an entropy feature reveals (0 without it), with
scaling taken from the first fold only.

### Real data (2026-10-04)

Gross bps per trade over all trades, for the pairs that differ only by the entropy features:

| | `spy_gao` | `etf_gao` | `etf_intraday` | `sp500_multihour` |
|---|---:|---:|---:|---:|
| `gbm` | −0.50 | +0.45 | +0.71 | +0.71 |
| `gbm_ent` | −0.23 | +0.50 | +0.47 | +0.85 |
| entropy's effect | +0.27 | +0.05 | −0.24 | +0.14 |
| `resnet1d` | −0.13 | +0.20 | +0.40 | not run |
| `resnet1d_ent` | −0.58 | +0.17 | +0.48 | not run |
| entropy's effect | −0.45 | −0.03 | +0.08 | |

No consistent sign, and every difference is far inside its models' intervals (roughly ±1 to ±2.7 bps).

The decision engine, models only → models plus the entropy features, net bps per opportunity after 2 bps
(share of samples traded):

| | bandit feedback | full feedback |
|---|---|---|
| `etf_intraday` | −0.18 (11%) → −0.35 (17%) | −0.28 (16%) → −0.80 (38%) |
| `etf_gao` | −0.27 (11%) → −0.38 (37%) | −0.27 (11%) → −0.67 (30%) |
| `sp500_multihour`, market-neutral | −0.20 (8%) → −0.79 (40%) | −0.19 (7%) → −0.60 (34%) |
| `sp500_multihour`, raw returns | −2.95 (63%) → −2.34 (60%) | −1.26 (57%) → −1.53 (39%) |

- **Uninformative inputs make the engine worse.** Given 13 entropy features with no signal, it finds patterns
  in the noise, trades two to five times as often and loses more. The raw S&P 500 rows still show the
  drift-chasing failure above, with or without entropy. (The models-only rows moved slightly from earlier runs
  because the new models' confidences are now inputs too.)
- **Missed opportunities: none.** No group has an interval above zero; choosing each group's best action in
  hindsight earns +0.06 (both ETF setups), +0.34 (S&P 500 raw) and +0.01 (market-neutral) bps per opportunity.

**Verdict on entropy.** Tested as a gate, as model inputs and as engine inputs, permutation entropy adds no
information to predictions from 5-minute prices in this data. That is consistent with the synthetic results:
what ordinal patterns of returns can see, the price window already shows, and the slow drift that matters at
multi-hour horizons they barely see at all.

## Advisor feedback and response (September 2026)

The feedback asked for:
- the model in mathematical terms, with its justification and references;
- its assumptions and predictions, and published empirical results to compare against;
- precise definitions of the target, the predictive variables and the entropy used.

It also noted that Fordham does not subscribe to TAQ, the data that would help with prediction windows of 30
seconds or less.

The response so far:
- **A formalization document**, kept outside this repo: targets, predictors, permutation entropy, weighted
  entropy and the trend statistic, the permutation test, assumptions, references, and the finite-sample bias
  table.
- **Horizons that 5-minute bars support.** `etf_multihour` (1-hour window → 2.5-hour hold) is configured but not
  run yet. `sp500_multihour` uses the same design and has been run.
- **A published baseline.** `spy_gao` and `etf_gao` reproduce the design of Gao, Han, Li and Zhou (2018). On
  2022–2026 IEX data the effect shows in direction on the ETFs (accuracy 0.517) but is too small to trade.

## Data notes

- **IEX coverage varies.** Among S&P 500 names, several mid-caps have bars in only 64–80% of 5-minute slots
  (for example AMP 63.7%, ALLE 74.6%, ALGN 75.9%). These names lose windows and gate readings.
- **Survivorship bias.** The S&P 500 list is today's constituents applied to 2020–2026 history. Point-in-time
  membership (WRDS/CRSP) would remove it.
- **Late listings start later,** for example ABNB in December 2020 and APP in April 2021.

## Explored but not built into the pipeline

- **Transfer entropy** (Schreiber 2000; symbolic version, Staniek and Lehnertz 2008), as a lead-lag measure
  between symbols. Validated in simulation: 0.183 in the true direction against 0.015 in reverse, peaking at the
  true lag, p = 0.01.
- **Kalman-filter state features:** proposed.
- **AWS architecture and costs:** designed. Colab Pro and GitHub are used instead.
- **IBKR:** a market-data recorder (quotes, trades, order-book depth) and a tick-to-bar converter are built, to
  run locally against TWS. A historical fetch through IBKR is agreed but not built.

## Progress log

| Date | Milestone |
|---|---|
| 2026-09-15 | Pipeline scaffold, GitHub repository, Colab notebook |
| 2026-09-16 | Refocus on SPY and the sector ETFs. Gate, evaluation and training rebuilt after the first sweep exposed the problems listed above |
| 2026-09-21 | Ordinal-pattern figure; confidence filtering and within-symbol gate scoring; incremental data top-up; IBKR capture tools |
| 2026-09-29 | 2.5-hour horizon configs after the advisor's feedback; S&P 500 universe with parallel fetching and screening |
| late September | Real-data runs of `spy_gao`, `etf_gao`, `etf_intraday` and `sp500_multihour` |
| 2026-10-01 | Decision engine, then full feedback and the missed-opportunities table. First real-data run: flat on the ETFs, collapsed into always-long on the S&P 500. Market-neutral mode built |
| 2026-10-02 | Market-neutral engine run on the S&P 500: no stock-specific signal survives cost; the engine stays flat |
| 2026-10-03 | Entropy became continuous features (overnight, 5-minute and hourly scales); trees and a ResNet1D with an entropy path; the engine reads the features. Found the 5-minute blind spot to slow drift |
| 2026-10-03 | Selective trading made the main case, with a risk-scaled check and a volatility placebo |
| 2026-10-04 | Entropy as features on real data: no effect as model inputs, and as engine inputs more trading and bigger losses |
| 2026-10-04 | Selective trading on real data: the confidence lead is volatility, not skill. Side finding on volatile moments, unchecked |
| 2026-10-04 | Diversification: breadth tools (independent bets, cross-sectional IC, market- and sector-neutral long-short books) and a 37-ETF cross-asset universe built and tested on synthetic markets; real-data runs pending |
| 2026-10-05 | Breadth results: 7–14 independent bets a day where we had 768 trades; with the market removed no S&P 500 model has stock-level skill, and the data could have detected half the IC a book needs |
| 2026-10-07 | Hold-out test built: train/trade universes, relative target, fixed-size books, Nasdaq-100 lists; protocol fixed before running |
| 2026-10-07 | Hold-out result: a null by the protocol (book net −0.53 bps), with the first significant ranking skill on unseen stocks (`gbm` IC +0.0141, t = 3.37) |
| 2026-10-07 | Skill checks (fixed tilt vs. timing, quarter by quarter) and a replication on the 418 training stocks out of time; rule fixed before running (committed 17:55; labelled 10-08 in the configs by mistake) |
| 2026-10-07 | Replication result: the hold-out skill does not replicate (416 stocks: IC +0.0005, t 0.2; timing −0.0055). On the 85 the IC is mostly a fixed tilt, timing +0.0019 (t 0.4). The reproduced 85 match the hold-out exactly |

## Next steps

1. Cost, only if wanted: measure real trading costs for these stocks (the IBKR recorder's quotes give spreads) and
   charge cost only when a position changes. The replication failed, so this would test whether a result that does
   not generalize would pay, which no longer needs to be a priority.
2. If the thesis adds an online learning agent: build it around a placebo and a replay-equivalence test first, since
   there is no surviving signal for it to adapt around. The question becomes whether online adaptation finds
   anything at all.
3. Optional, about an hour: check the volatile-moment side finding (market-neutral, without April 2025, and with
   a higher cost in volatile moments) before it goes into the thesis even as a side note.
4. Run `etf_multihour`, the horizon the advisor pointed toward.
5. Optional: relative input features (each stock's window measured against its peers), and the deep models on the
   relative target (a long GPU run). The tree model's relative-target skill did not replicate, so a deep model is
   unlikely to turn it into one.
6. Then stop adding complexity and write up: a careful null with its methodological findings (leakage fixes,
   inference by day, the drift blind spot of permutation entropy, the engine's failure modes).
7. Count gate approvals as episodes, not days.
8. Optionally, run the deep models on the S&P 500 (`--archs resnet1d cnn1d`), a long run.
9. Fix wording: the S&P 500 gate table still says "ETF".
10. Use point-in-time S&P 500 membership to remove survivorship bias.
