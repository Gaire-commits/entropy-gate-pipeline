# Entropy Gate Pipeline

Predictability screening before prediction, on SPY, the 11 Select Sector SPDR ETFs and
the S&P 500. Permutation entropy reads the ordinal structure of each symbol's recent
returns. It started as a nightly pass/fail gate and is now a set of continuous features
that the models and the decision engine weigh for themselves. A ladder of rules,
gradient-boosted trees and deep networks predicts intraday direction, a reinforcement-
learning engine decides whether to trade, and every result is judged on returns after
trading costs, with uncertainty measured in days.

```
5-minute bars ─┬─ deep encoder (ResNet1D on the price window) ──┐
               ├─ entropy features (overnight + intraday, ─────┼─ models: gbm, gbm_ent, resnet1d,
               │   5-minute and hourly scales)                  │          resnet1d_ent, rules ...
               └─ window summary statistics ───────────────────┘                │
                                                     confidences + entropy ─ RL engine ─ short / flat / long
```

**Results so far and progress:** [RESULTS.md](RESULTS.md).

## Running it

**On Google Colab (recommended):** open [`notebooks/colab.ipynb`](notebooks/colab.ipynb).
It clones this repo, keeps downloaded data and results on Google Drive, and reads
credentials from Colab's Secrets panel. Every step resumes after a disconnect.

**Locally:**

```bash
pip install -r requirements.txt
cp .env.example .env                 # add your Alpaca keys
python -m pytest tests/ -q           # no credentials needed
python scripts/smoke_test.py         # synthetic data with planted signals

python scripts/fetch_data.py         # SPY + 11 sector ETFs, 5-min bars; reruns only add new days
CONFIG=configs/spy_gao.yaml
python scripts/run_screen.py --config $CONFIG
python scripts/sweep.py      --config $CONFIG
python scripts/summarize.py  --config $CONFIG   # -> outputs/spy_gao/summary.md
```

## Experiments

| Config | Universe | Signal → hold | Question |
|---|---|---|---|
| `configs/spy_gao.yaml` | SPY | first half-hour → last half-hour | Does the published intraday momentum effect (Gao et al. 2018) show up? A real-data check that the pipeline can find something known. |
| `configs/etf_gao.yaml` | 12 ETFs | first half-hour → last half-hour | Does it hold across sectors? |
| `configs/etf_intraday.yaml` | 12 ETFs | 4-hour window → 1-hour hold | Can learned features beat simple rules? |
| `configs/etf_multihour.yaml` | 12 ETFs | 1-hour window → 2.5-hour hold | Same question at a horizon 5-minute bars fully support (no TAQ needed) |
| `configs/sp500_multihour.yaml` | S&P 500 (~503 stocks) | 1-hour window → 2.5-hour hold | Does the gate find more structure in individual names than in diversified ETFs? |
| `configs/cross_asset_multihour.yaml` | 37 ETFs in 7 asset classes | 1-hour window → 2.5-hour hold | Are signals in bonds, commodities and currencies closer to independent bets than another equity universe? |
| `configs/ndx_holdout.yaml` | train: 418 S&P 500 stocks; trade: 85 Nasdaq-100 members they never saw | 1-hour window → 2.5-hour hold, relative target | Does a model of which stocks beat their peers carry over to stocks it was not trained on? |

Each run answers the research questions from the same predictions:

| | Question | Where it shows up |
|---|---|---|
| Q1 | Does the gate pick better days? | `summary.md` gate table: the same predictions on approved vs. other days, with a 95% interval on the difference |
| Q2 | Do deep models beat simple rules and a linear model? | `summary.md` main table: `always_up`, `momentum_day`, `momentum_window`, `logreg`, then CNN1D, ResNet1D, InceptionTime |
| Q4 | Does an image view of the window help? | ResNet2D (Gramian Angular Field images) vs. ResNet1D, same stages and filters |
| Q5 | Does any edge survive costs? | net return at each cost level, and the breakeven cost |
| Q6 | Do entropy features help, as inputs rather than a gate? | pairs that differ only by the entropy features: `gbm_ent` vs `gbm` (trees), `resnet1d_ent` vs `resnet1d` (deep), and the engine with vs. without them (`engine.md`) |

Two further checks come free with the same predictions, no retraining:

- **Confidence filtering**: results using only the most confident predictions (`evaluation.coverage_levels`).
  Each fold's cutoff comes from earlier folds only, so nothing uses the future.
- **Within-ETF gate comparison**: the Q1 difference after subtracting each ETF's own average. The gate refuses
  whole ETFs that move only a few cents a bar, so the raw split partly compares expensive ETFs against cheap ones.

## Breadth: how many independent bets are there?

Trades placed at the same moment share the market's move, and our models are almost always long (79–98% of
the S&P 500 trades), so hundreds of trades a day were closer to one bet on the market than to hundreds of
bets. The fundamental law of active management (Grinold and Kahn) says why it matters: information ratio ≈
IC × √breadth, and breadth counts *independent* bets.

```bash
python scripts/breadth.py --config configs/sp500_multihour.yaml    # no retraining; reads the saved predictions
```

For each model it reports, with day-block intervals:

- **Effective independent bets per day** for the same model used three ways (every signal, its top 10%, a
  long-short book): n / (1 + (n − 1)ρ), where ρ is the correlation between a day's positions, read from how
  much more the day's average moves than independent positions would. About 1 is one bet on the market.
- **Cross-sectional IC**: at each date and bar, the rank correlation between the score and the realized return
  across stocks, averaged over days. The market move is the same for every stock at a moment, so it cancels,
  and every stock contributes, which makes it the most powerful test of stock-level skill the data allows. The
  sector-neutral version also cancels sector moves, so a model that only knows which sector will move scores
  nothing.
- **A long-short book**: long the top 10% and short the bottom 10% of the score at each moment (optionally within
  each sector), market-neutral by construction, with the cost charged per position.
- **What the data can rule out**: the smallest IC it would detect (2.8 standard errors), against the IC a
  long-short book needs to pay its cost, cost ÷ (spread across stocks × average |z| of the picks). If the first is
  smaller, a null result rules out a profitable signal of this kind; if not, it does not.
- **How alike the models are**: their scores' average correlation and the number of independent models it
  amounts to. Models trained on the same inputs and labels are close to one model.

Asset-level breadth comes from the data: `configs/cross_asset_multihour.yaml` runs the same pipeline on 37 ETFs
in 7 asset classes (`universe/cross_asset.csv`; the `sector` column is the asset class), whose drivers differ.
Its caveats are in the config: IEX prints the thinner ETFs rarely, so check each symbol's bar coverage, and
round-trip costs differ by class.

Controls (`tests/test_crosssection.py`): on synthetic markets with a market factor, sector factors and an
almost-always-long model, the IC finds a planted stock-level skill (t > 4) that the directional test cannot
see (its interval spans zero), invents nothing when there is no signal, and the sector-neutral IC removes a
signal that is only knowing the sector but keeps real skill; the long-short book has zero net exposure; the
effective-bets estimator recovers a known correlation (ρ = 0, 0.1, 0.3, 0.7 within 0.03).

## Hold-out: train on one universe, trade another

`configs/ndx_holdout.yaml` trains on the S&P 500 without the Nasdaq-100's members (418 stocks) and scores only
the 85 Nasdaq-100 members in the S&P 500, which the models never see in training or validation. The other 15
Nasdaq-100 members have no comparable history and are left out.

```bash
python scripts/build_universe.py --index nasdaq100       # universe/nasdaq100.csv (today's list)
python scripts/build_universe.py --holdout nasdaq100     # universe/sp500_ex_nasdaq100.csv, nasdaq100_in_sp500.csv
python scripts/sweep.py   --config configs/ndx_holdout.yaml
python scripts/breadth.py --config configs/ndx_holdout.yaml   # ranks the 85 and holds 12 long, 12 short
```

- **Roles** (`data.train_universe`, `data.trade_universe`): training and validation rows come only from the
  training universe, test rows only from the trading one (`src/holdout.py`). Without them every symbol does
  both, as before.
- **Relative target** (`features.target: relative`): did the stock beat the average of its own universe at that
  date and bar? Training stocks are compared with training stocks and held-out stocks with held-out stocks, so
  the market's move cancels from the label and neither universe's average leaks into the other. `ret` becomes
  the return against peers everywhere downstream; always-long earns exactly zero by construction.
- **Fixed-size books** (`evaluation.book_per_side`, `breadth.py --n-per-side`): the top and bottom n names at each
  moment; the sector-balanced version gives each sector picks in proportion to its size.
- **The rules for a finding are in the config, written before any result**: for the same model, a
  cross-sectional IC with t ≥ 3 and a 12-a-side book whose net 95% interval is above zero.

It is a hold-out across stocks, not across time: both universes live through the same dates. And today's
Nasdaq-100 list holds the stocks that rose enough to be in it, which flatters anything that leans long.

Controls (`tests/test_holdout.py`): no held-out stock in any training or validation row, and held-out returns
never change a training label; models trained on synthetic stocks find a shared stock-level signal in stocks
they never saw (IC above 0.2) and find nothing when only the training stocks carry it.

Two follow-ups for a positive IC, both on saved predictions:

```bash
python scripts/skill_checks.py --config configs/ndx_holdout.yaml        # fixed tilt or timing? quarter by quarter, time of day
python scripts/breadth.py      --config configs/ndx_replication.yaml --only universe/sp500_ex_nasdaq100.csv
```

`skill_checks.py` splits an IC into a *static* part (each stock ranked by its average score in earlier quarters, a
fixed preference that survivorship in today's lists can reward) and a *timing* part (the score minus that average,
against returns minus each stock's own average). `configs/ndx_replication.yaml` trains the same models on the same
418 stocks and scores everything, so the 418 can be judged out of time (`--only` scores a group separately); its
rule for a replication is written in the config.

Result: it did not replicate. On the 416 scored training stocks the `gbm` IC is +0.0005 (t = 0.2) and the timing IC
is −0.0055; on the 85, timing is +0.0019 (t = 0.4), so the hold-out's IC is mostly a fixed tilt. See RESULTS.md.

## Selective trading (the main case)

`scripts/selective.py` trades only each model's strongest signals, the top 20%, 10%, 5% or 2% by
confidence, with each quarter's cutoff set from earlier quarters only. Breadth keeps it frequent:
the top 10% is about one trade a day on 12 ETFs and about 60 a day on the S&P 500. It reports
trades per day, hit rate, net bps per trade and per day, and in how many quarters it made money.

```bash
python scripts/selective.py --config configs/sp500_multihour.yaml    # no retraining; reads predictions and bars
```

It also asks whether the strongest signals are skill or just volatility. A model that is most
confident when the market is most volatile earns more bps on its confident trades simply because
every move is bigger. So each trade's return is also divided by its symbol's trailing volatility
times the square root of the hold (*risk-scaled*), and a placebo keeps the same share of signals
ranked by trailing volatility instead of confidence. Skill means the confident trades earn more per
unit of risk than the placebo. Compare in risk-scaled terms, not bps: in a test where a model was right
66% of the time everywhere, the volatility placebo earned more bps (+40.8 against +32.5) because it
picked the biggest moves, while risk-scaled return showed the skill (49 against 25).

## Decision engine

`src/engine.py` and `scripts/engine.py` train a reinforcement-learning policy on the saved
walk-forward predictions. Per sample it chooses short, flat or long and is paid
`position x return - cost x |position|`. Staying flat pays exactly 0, which is the reference.

```bash
python scripts/engine.py --config configs/etf_intraday.yaml    # after scripts/sweep.py
# --feedback bandit|full|both (default both); --signal resnet1d grades the missed-opportunities table by one model
python scripts/engine.py --config configs/sp500_multihour.yaml --neutral   # market-neutral, see below
```

It learns online, fold by fold, from earlier feedback only, with a policy gradient (REINFORCE). It is
compared with the best fixed rule chosen from the past. It runs with and without the entropy information
as input, which is the direct test of whether the entropy signal helps an agent that can learn from
feedback. That input is the continuous entropy features when the sweep saved them (scaled with statistics
from the first fold, which the engine only observes), and the old gate's pass flags for older predictions.
It also runs with two kinds of feedback:

- **bandit**: only the reward of the action it took. Flat pays 0 and says nothing about what a trade would
  have paid, so once the policy settles on flat it stops seeing the opportunities it misses.
- **full**: all three actions scored on every past sample, including the trades it skipped. Prices do not
  react to our trades, so in a backtest those rewards are exact. Live, the fill a skipped trade would have
  got is unknown, and bandit feedback is the honest one.

Missed opportunities are measured by situation, not by trade: `engine.md` groups samples by gate verdict and
signal strength and shows each group's best fixed action (picked in hindsight, with an interval) next to what
each engine earned there. Training on every individual missed trade is the wrong fix: with noise much larger
than the cost, almost every sample looks like a missed trade in hindsight, and a policy taught to copy the
hindsight-best action traded everything and lost about 2 bps per opportunity on pure noise.

**Market-neutral mode (`--neutral`).** On the real S&P 500 data the engine as first built failed: it traded
76-98% of samples and lost 1.2-3.0 bps per opportunity, and under full feedback its P&L matched always-long's to
the hundredth in 8 of 9 folds. Every stock sampled at the same date and bar shares that moment's market move, and
the market's direction flips from quarter to quarter, so one quarter's drift looked like tens of thousands of
confirmations of "go long" and did not hold the next quarter. Learning from all past quarters does not fix it
(there are only about 14 independent quarters), so `--neutral` subtracts the cross-sectional mean at each date and
bar from every return (the reward) and every model confidence (the input). On synthetic data with a drifting
market it stays flat when there is no stock-specific edge and finds one when planted, while the raw engine loses
on noise and finds the edge only erratically. It describes a long-short book with costs on the stock leg only, so it is
optimistic by the hedging cost, and it needs a wide universe: it refuses the 12 ETFs, which are mostly the
market.

Because prices do not react to our trades, replay is a faithful simulator. That also means this is a
contextual bandit (one-step RL): it has no inventory, holding limits or position-dependent costs. Those
need a stepwise environment and are the natural next extension.

Controls in `tests/test_engine.py`: under both kinds of feedback it must learn a planted edge and beat
trading every signal, and on pure noise it must learn to stay out (net 0 against about -2 bps for trading
every signal). Full feedback must also learn an edge barely above cost, which bandit feedback misses on one
of two seeds, and find an edge planted only on gate-approved samples, which it does only when given the
gate's verdict. Two traps found while building it: a flat-leaning initialization saturates the softmax and
the agent never trades again, and with bandit feedback an edge barely above cost is sometimes not learned at
all, which is the safe failure.

## Online agent (streaming bars)

`src/online.py` takes the engine off the quarterly folds: an agent that learns while the bars arrive, one
bar at a time, as it would live.

```bash
python scripts/online_replay.py --config configs/online_sp500.yaml
python scripts/online_replay.py --config configs/online_sp500.yaml --only universe/nasdaq100_in_sp500.csv
```

- **The stream.** `StreamFeaturizer` sees each bar once and keeps only short ring buffers. At each signal bar
  it cuts the same windows as the batch pipeline. A trade's return is attached only when its exit bar
  arrives, 31 bars later, and the agent learns from it then.
- **Rewards.** Market-neutral: each stock against the average at that moment, 2 bps per position.
- **Exploration.** Exploration is set by the policy's entropy H(π) = −Σ π log π, not a fixed bonus. The
  bonus weight adapts so the entropy tracks a target (0.5 nats; ln 3 is uniform over short / flat / long).
  A Page-Hinkley drift detector raises the target for a while when the greedy policy's results drop.
- **Three books per agent.** *greedy* is the most likely action, what would be deployed. *explore* samples
  from the policy and pays for exploring. *frozen* is the greedy policy at the end of warm-up, never
  updated, which shows whether continuing to learn helps.
- **Placebo worlds.** Each run is repeated with each moment's returns shuffled across its stocks: same
  features, volatility and timing, nothing to learn. The rule for a finding is fixed in the config.
- **Paper only.** Positions are paper positions; nothing here places an order.

Controls in `tests/test_online.py`:

- **Same as batch.** The stream reproduces every batch sample's features and returns with missing bars,
  missing days and an early close. Cutting off the future changes nothing before the cut, and the script
  repeats this check on real bars before it runs the agent.
- **Nothing early.** A return is read only at its outcome event, exactly `embargo + horizon` bars after the
  decision.
- **Signal and noise.** The agent learns a planted cross-sectional signal under both kinds of feedback and
  makes nothing on noise or in the placebo world.
- **Entropy matters.** When a planted signal flips sign mid-stream, the agent with entropy control earns it
  again (above +1 bps per opportunity over the last 300 moments). Without it the agent stays below +0.5,
  under full feedback too: the softmax saturates and the policy stops moving. Under bandit feedback, flat
  then pays exactly zero and teaches nothing.

## E1 and E2: settling the open questions

Two pre-registered experiments; each rule is in its config, and `tests/test_frozen.py` fails if a
pre-registered config is edited.

**E1, the volatile-moment lead.**
```bash
python scripts/volatility_lead.py --config configs/e1_volatility.yaml
```
`src/volatility_lead.py` tests the +15.21 bps found at the 2% most volatile S&P 500 moments once, without a
model.
- **The rule:** go long the top 2% by trailing volatility, with each month's cutoff set from earlier months.
- **Untouched data:** it is scored on 2020–2022, which nothing in this project had scored.
- **Uncertainty:** intervals resample blocks of 5 trading days, because volatile days cluster.
- **Costs:** a flat 2 bps, and a stress cost that rises with volatility.
- **Market or stock:** a market-neutral version shows which one the gain belongs to.
- **Not one episode:** the test quarters are rescored without the largest volatile episode, without the
  10 biggest days, and without April 2025.

**E2, model or signal?**
```bash
python scripts/capacity.py --config configs/e2_capacity.yaml --part curve
python scripts/capacity.py --config configs/e2_capacity.yaml --part planted
```
`src/capacity.py` runs two tests on the replication's setup:
- **Learning curve and capacity ladder:** trees trained on 126 to 504 days, with 4 to 63 leaves, all
  scored on the same rows.
- **Planted edges:** the real returns plus an edge built from the model's own inputs, at ICs from 0.005 to
  0.02 (break-even 0.014). The question is whether the same pipeline finds it.

The tree size became a setting for E2 (`model.gbm_leaves`, `model.gbm_trees`). Its defaults are what
every earlier run used, and a test checks the predictions are unchanged.

Controls in `tests/test_volatility_lead.py` and `tests/test_capacity.py`:
- **E1 uses only the past:** cutoffs and medians come from earlier months.
- **E1 sorts planted cases correctly:**
  - no effect: not replicated;
  - an effect in every period: replicated and stock-level;
  - one episode only: not replicated;
  - a market-wide rebound: replicated, not stock-level.
- **E2 cells differ only where they claim to.**
- **The planted edge uses only the inputs and hits its target IC.**
- **The trees find a planted edge end to end, and find nothing when none is planted.**

## Larger universes

A config's `universe` can be a list of symbols or a path to a CSV built by
`scripts/build_universe.py`:

```bash
python scripts/build_universe.py --index sp500      # -> universe/sp500.csv (today's constituents)
python scripts/fetch_data.py  --config configs/sp500_multihour.yaml --workers 4
python scripts/run_screen.py  --config configs/sp500_multihour.yaml --workers 0   # 0 = all CPU cores
```

Each symbol's gate reading depends only on its own history, and its shuffles are
seeded from its own name, so parallel screening gives exactly the sequential result
(a test checks this). A universe built from today's constituents carries
survivorship bias against historical data: names that left the index are missing.

## Entropy as features

The hard gate approved about as often as pure noise and threw away everything about a
reading except one bit, so the same ordinal statistics now enter as continuous inputs
(`src/entropy_features.py`), all known before the trade:

- **overnight**, from the nightly reading below: entropy and weighted entropy as z-scores
  against shuffled copies, the share of monotone patterns and its excess over the shuffles,
  statistical complexity, the share of tied values and cents per bar;
- **intraday**, at the signal bar: permutation entropy (m = 3, 4) and the monotone share of
  the last 60 five-minute returns, and the same at m = 3 on hourly returns over 5 sessions.

**The blind spot behind the hourly features.** Ordinal patterns of 5-minute returns are nearly
blind to a slowly varying drift, the kind of trend multi-hour momentum relies on: a drift
that barely changes over three bars shifts all three returns alike and leaves their ranks
as noise would. For a Gaussian process the monotone share at m = 3 depends on the
autocorrelations only through (2ρ1 − 1 − ρ2) / (2(1 − ρ1)), so a drift with ρ1 ≈ ρ2 reads
as noise however strong it is. In simulation, 5-minute patterns separate trending from
noise spells by 0.07–0.17 standard deviations, hourly ones by about 0.9, and a plain
variance ratio by about 2 (`tests/test_entropy_features.py` pins this down).

**Making the deep model use them.** `resnet1d_ent` has ResNet1D's body plus an entropy path. With
the window mostly noise, the body memorizes it within an epoch or two, and early stopping froze
a plain concatenation before it used the entropy inputs at all (0.50 accuracy where 0.75 was
available). So the linear entropy path is fitted first as a logistic regression, the classifier
starts at zero, and that fit is the candidate the joint training has to beat.

Trees (`gbm`, `gbm_ent`) use scikit-learn's histogram gradient boosting on summary statistics
of the window, with the tree count chosen on the validation fold. They are deterministic, so the
sweep runs them once.

## How the gate works

Each evening, for every ETF, the gate takes the last 5 sessions of 5-minute log
returns and compares their ordinal patterns (Bandt & Pompe) against 99 shuffled
copies of the same returns. Shuffling keeps every value and destroys only the
order, so fat tails and a few giant moves cannot pass for structure.

Three statistics are recorded for every reading:

| Statistic | Passes when | Problem |
|---|---|---|
| `entropy` | permutation entropy is below the shuffles | also fires on zig-zags from bid-ask bounce and penny ticks, which nobody can trade |
| `entropy_weighted` | weighted permutation entropy is below the shuffles | also fires on volatility clustering |
| `trend` (default) | steady up-runs and down-runs appear more often than in the shuffles | — |

A reading also fails if the window is mostly repeated prices (`stale`) or moves
less than about 6 cents a bar (`tick_limited`), where rounding to the cent alone
creates patterns. The gate is absolute: on a day when nothing passes, nothing is
approved. Under pure noise about 5% of readings still pass, which is why Q1 is
judged by what happens on approved days rather than by how many there are.

A reading computed at Monday's close applies to Tuesday, exactly as a nightly
job would run live.

## IBKR capture (local, optional)

`notebooks/ibkr_capture.ipynb` records quotes, trades and order-book depth from a TWS paper account and turns
the capture into the same 5-minute bars the pipeline reads. It runs on your Mac (TWS has to), not on Colab,
and it only reads market data: nothing places an order.

```bash
pip install ibapi
jupyter notebook notebooks/ibkr_capture.ipynb     # run it 09:30-16:00 ET on a weekday
```

Captures are written to `data/ibkr/` every 5 seconds. Bars are built from the quote **midpoint**, which does not
bounce between bid and ask the way trade prices do, so it is a cleaner series than the Alpaca trade bars but not
the same one: compare the two feeds on a shared day before trusting one to stand in for the other. The gate needs
5 sessions of returns plus the one it scores, so the first reading takes 6 complete sessions of capture.

Depth needs a separate paid subscription per exchange, and IBKR allows only a few depth streams at once.
The first version of this recorder had three faults, all fixed in `src/ibkr_capture.py`: depth was requested as
`SMART` with `isSmartDepth=False` (IB error 10092, so nothing was recorded), the `operation` column was not
saved (so a deleted level looked like an updated one and the book could not be rebuilt), and the in-memory book
stored levels in a dict although IB positions shift on every insert and delete.

## Layout

| Path | Role |
|---|---|
| `src/entropy.py` | Permutation entropy, complexity, and the shuffle test |
| `src/entropy_features.py` | Continuous entropy features, overnight and intraday |
| `src/screening.py` | Nightly gate readings and pass/fail rules |
| `src/data.py` | Alpaca download with caching, and the fixed 5-minute session grid |
| `src/features.py` | Causal channels, windowing, normalization, GAF/MTF encoders |
| `src/dataset.py` | Samples with gate verdicts attached; walk-forward splits |
| `src/baselines.py` | Rules that need no training |
| `src/models.py` | logreg / CNN1D / ResNet1D / InceptionTime / ResNet2D / ResNet1D + entropy |
| `src/ml.py` | Gradient-boosted trees, with and without the entropy features |
| `src/training.py` | Reproducible training with early stopping |
| `src/stats.py` | Trade scoring and day-block bootstrap intervals |
| `src/experiment.py` | Resumable sweep and the results report |
| `src/synthetic.py` | Bars with known structure for tests |
| `src/engine.py` | RL decision engine: policy, online walk-forward learning, baselines |
| `src/selective.py` | Selective trading: past-only cutoffs, frequency, risk-scaled return, volatility placebo |
| `src/crosssection.py` | Breadth: effective independent bets, cross-sectional IC, long-short books, model similarity |
| `src/holdout.py` | Training and trading universes, the relative target, and the hold-out split |
| `src/volatility_lead.py` | E1: the volatile-moment lead without a model, on untouched years, with block intervals and stress costs |
| `src/capacity.py` | E2: learning curve, capacity ladder and planted edges |
| `src/online.py` | Online agent: streaming featurizer, delayed rewards, entropy-controlled exploration, drift detector, placebo replay |
| `universe/` | Saved constituent lists for larger universes |
| `src/ibkr.py` | IBKR ticks to 5-minute bars, and depth replay |
| `src/ibkr_capture.py` | TWS recorder for quotes, trades and depth |

## Design notes

**Same seed, same result.** Training is deterministic, and every model runs with 3
seeds. In the first real sweep, before this, two identical reruns of one model
differed by about as much as the five architectures differed from each other.

**The trend has to stay visible.** Features are standardized with statistics from
the training fold only. The earlier per-window z-score subtracted each window's
mean, which set every window's cumulative return to zero: the models literally
could not see whether price went up. On planted data that cost logistic
regression 14 points of accuracy.

**Uncertainty is measured in days.** Trades on the same day share the market's
move, so intervals come from resampling whole days. Treating thousands of same-day
trades as independent would make every interval several times too narrow.

**The test set is never filtered by outcome.** Training may skip moves under 5 bps
as noise; the test set keeps every sample, because which moves turn out small is
only known afterwards.

**Data tops up, it doesn't restart.** `end: "today"` in a config makes every fetch download only the days since the
last cached one and merge them in. A session still in progress is dropped, so running mid-day never leaves a
half day that looks like a complete one. Alpaca's free IEX history begins in late July 2020, so that is where
the configs start. Downloaded data lives in `data/` (Google Drive on Colab) and is never committed.

**Time is fixed to the clock.** IEX only prints a bar when a trade happens on IEX.
Missing bars are restored as gaps on a fixed 5-minute grid, so "the last half-hour"
always means 3:30–4:00 and a gap drops the affected window instead of shifting it.

**Intraday means intraday.** A window, its execution lag and its hold must fit in
one session, or sample construction raises.

**Positive controls.** `smoke_test.py` plants signals in both setups and checks that
the rules and models recover them. Without that, a null result on real data is
indistinguishable from a broken pipeline.

## Status

Built and tested: data, gate, features, models, sweep, report, decision engine, online agent, E1 and E2
(266 unit tests plus the smoke test). Run on real data for SPY, the 12 ETFs and the S&P 500;
`etf_multihour` not yet. Findings are in [RESULTS.md](RESULTS.md). The live path (IBKR
execution, risk firewall) and position sizing are designed, not built.

Known limits: IEX volume is a sample of the consolidated tape; Kalman state
features are proposed, not built.
