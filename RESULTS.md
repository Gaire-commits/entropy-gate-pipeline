# Results and progress

Last updated 2026-10-01. Numbers come from each experiment's `outputs/<experiment>/summary.md`, which
lives on Google Drive rather than in this repo. The runs finished in late September 2026.

## In short

- **Nothing beats trading costs.** Four experiments: SPY, the 12 sector ETFs at two horizons, and 503
  S&P 500 stocks. Every model and rule loses money after a 1–2 bps round-trip cost, and no gross return has
  a 95% interval above zero.
- **Deep models do not beat simple ones.** No deep model (CNN1D, ResNet1D, InceptionTime, ResNet2D) is the best
  row in any experiment. Where one edges past logistic regression, the margin is under 0.2 bps, far inside the
  intervals.
- **The entropy gate does not pick better days.** The default trend gate approves about as often as it would
  on pure noise. Of 60 approved-vs-other comparisons, 5 have intervals that exclude zero, about 3 would by
  chance, and 2 of the 5 point the wrong way.
- **One unconfirmed lead.** Trading only a model's most confident predictions raises the return per trade in
  4 of 10 model runs, most strongly for S&P 500 logistic regression. Every interval includes zero and accuracy
  barely moves, so it may track volatility rather than skill.
- **The measurement works.** On synthetic data the gate passes planted trends and refuses zig-zags and tick
  noise, and the models recover planted signals. The null on real data is a finding, not a broken pipeline.
- **The decision engine (RL) finds nothing either.** On the ETFs it learns to stay flat. On the S&P 500 the
  first version collapsed into always-long, a flaw in the engine, not a market finding. A market-neutral
  version is built and its real-data run is pending.

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
| `sp500_multihour` | 503 stocks | 1-hour window → 2.5-hour hold | 2 bps | run, rules and logreg only |
| `etf_multihour` | 12 ETFs | 1-hour window → 2.5-hour hold | 2 bps | not run yet |

## Does anything beat costs?

Gross basis points per trade with 95% intervals, and net at each experiment's cost.

**`spy_gao`** (net at 1 bps)

| model | params | accuracy [95% CI] | gross bps/trade [95% CI] | net |
|---|---:|---|---|---:|
| always_up | 0 | 0.507 [0.474, 0.540] | +0.29 [−1.21, +1.63] | −0.71 |
| momentum_day | 0 | 0.508 [0.476, 0.541] | +0.56 [−0.78, +1.95] | −0.44 |
| logreg | 50 | 0.503 [0.470, 0.533] | −0.29 [−1.81, +1.12] | −1.29 |
| resnet1d | 505,730 | 0.490 [0.458, 0.521] | −0.13 [−1.54, +1.29] | −1.13 |

**`etf_gao`** (net at 2 bps)

| model | params | accuracy [95% CI] | gross bps/trade [95% CI] | net |
|---|---:|---|---|---:|
| always_up | 0 | 0.494 [0.473, 0.516] | −0.54 [−1.88, +0.76] | −2.54 |
| momentum_day | 0 | 0.517 [0.502, 0.531] | +0.52 [−0.33, +1.40] | −1.48 |
| logreg | 50 | 0.502 [0.489, 0.516] | +0.58 [−0.21, +1.47] | −1.42 |
| resnet1d | 505,730 | 0.505 [0.493, 0.519] | +0.20 [−0.50, +0.92] | −1.80 |

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

**`sp500_multihour`** (net at 2 bps, 673,694 trades)

| model | params | accuracy [95% CI] | gross bps/trade [95% CI] | net |
|---|---:|---|---|---:|
| always_up | 0 | 0.506 [0.495, 0.516] | +1.03 [−1.61, +3.84] | −0.97 |
| momentum_day | 0 | 0.496 [0.490, 0.501] | −0.87 [−2.28, +0.40] | −2.87 |
| momentum_window | 0 | 0.496 [0.492, 0.500] | −0.63 [−1.51, +0.23] | −2.63 |
| logreg | 98 | 0.503 [0.496, 0.510] | +0.86 [−0.95, +3.01] | −1.14 |

What the tables say:

- **The best row in each experiment is a rule or logistic regression.** No deep model is best anywhere.
- **Image view (Q4):** ResNet2D on Gramian Angular Field images, with 914,690 parameters, earns +0.32 against
  ResNet1D's +0.40 on the same windows. The image encoding adds nothing.
- **The one accuracy interval clear of 50%** is half-hour momentum across the ETFs: 0.517 [0.502, 0.531]. The
  last half-hour moves in the direction of the return from the previous close to 10:00 slightly more often
  than not, the direction of the published intraday momentum effect (Gao, Han, Li and Zhou 2018). It is worth
  +0.52 bps a trade, about a quarter of the cost.
- **Single stocks lean the other way.** On the S&P 500 both momentum rules are right 49.6% of the time, a hint
  of short-term reversal. Fading them would earn under 1 bps gross, still below cost.
- **At the longer horizons, simply holding long beat every model,** and that is mostly market drift. On the
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

The same predictions, split by whether the gate approved the day: 60 comparisons (model × statistic ×
experiment). Five have 95% intervals that exclude zero, where chance alone would give about 3, and two of the
five point the wrong way:

| Experiment | Model | Gate | Approved − other, bps [95% CI] |
|---|---|---|---|
| `spy_gao` | logreg | trend | −9.57 [−17.47, −2.85] |
| `spy_gao` | logreg | weighted entropy | +4.48 [+0.66, +8.34] |
| `etf_gao` | logreg | entropy | +1.97 [+0.04, +3.99] |
| `etf_intraday` | always_up | weighted entropy | −1.47 [−3.07, −0.03] |
| `etf_intraday` | inceptiontime | entropy | +1.67 [+0.01, +3.29] |

- **The default trend gate:** 1 of its 20 comparisons excludes zero, and that one is negative.
- **Within each symbol:** the gate refuses whole low-priced symbols, so the raw split partly compares one
  symbol with another. After subtracting each symbol's own average, 2 of 48 comparisons exclude zero, about what
  chance gives (2.4).
- **The best-looking ETF lead did not replicate.** At 4h→1h on the ETFs, trend-approved days looked better for
  momentum_window (+2.47 [−0.19, +5.00]) and resnet1d (+2.02 [−0.42, +4.61]). On the S&P 500, momentum_window
  shows +0.22 [−1.17, +1.47].

## Trading only confident predictions

Each fold's confidence cutoff is set from earlier folds only. Gross bps per trade at each share of trades kept:

| Experiment, model | 100% | 50% | 20% | 10% [95% CI] | net at 10% |
|---|---:|---:|---:|---|---:|
| `sp500_multihour`, logreg | +0.89 | +1.90 | +3.96 | +6.15 [−1.56, +14.80] | +4.15 |
| `etf_intraday`, resnet1d | +0.47 | +1.06 | +1.70 | +2.36 [−0.27, +4.84] | +0.36 |
| `etf_gao`, logreg | +0.66 | +0.75 | +1.15 | +1.74 [−2.09, +5.88] | −0.26 |
| `etf_intraday`, cnn1d | +0.09 | +0.18 | +0.23 | +0.58 [−1.96, +2.79] | −1.42 |
| `etf_intraday`, resnet2d | +0.39 | +0.32 | +0.42 | +0.34 [−1.24, +1.90] | −1.66 |
| `etf_gao`, resnet1d | +0.27 | +0.57 | +0.47 | −0.02 [−3.92, +4.23] | −2.02 |
| `etf_intraday`, logreg | −0.18 | −0.07 | −0.02 | −0.53 [−4.47, +3.90] | −2.53 |
| `spy_gao`, resnet1d | −0.61 | −0.72 | −1.69 | −0.60 [−6.67, +5.21] | −1.60 (at 1 bps) |
| `etf_intraday`, inceptiontime | −0.11 | −0.39 | −0.40 | −1.22 [−3.95, +1.23] | −3.22 |
| `spy_gao`, logreg | +0.33 | −1.26 | +0.44 | −2.17 [−10.39, +5.76] | −3.17 (at 1 bps) |

The 100% column differs slightly from the main tables because early folds without enough history to set a
cutoff are dropped at every level. The realized share kept also differs a little from the target; for
example, the S&P 500 logreg 10% level keeps 8%.

Return rises with confidence in the top four rows and shows no consistent rise in the other six. It is not
an edge yet:

- **Every interval includes zero.**
- **The levels are nested:** the 10% trades are part of the 20% trades, so each row is one result, not four.
- **Accuracy barely moves.** S&P 500 logreg goes from 0.503 to 0.512, so the gain comes from bigger moves, not
  more correct calls. That fits a model that is most confident on volatile days, when every move is larger.
- **The next check:** re-score the trades in units of each day's volatility. If the slope survives, it is worth
  pursuing.

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
  That mode is built (`scripts/engine.py --neutral`); its S&P 500 result is pending.

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

## Next steps

1. Run the market-neutral engine on the S&P 500 (last cell of Colab section 11) and record the result here.
2. Run `etf_multihour`, the horizon the advisor pointed toward.
3. Re-score the confidence filter in units of each day's volatility before citing it.
4. Count gate approvals as episodes, not days.
5. Optionally, run the deep models on the S&P 500 (`--archs resnet1d cnn1d`), a long run.
6. Fix wording: the S&P 500 gate table still says "ETF".
7. Use point-in-time S&P 500 membership to remove survivorship bias.
