# Entropy Gate Pipeline

Predictability screening before prediction, on SPY and the 11 Select Sector SPDR ETFs.
A nightly gate reads the ordinal structure of each ETF's recent returns and decides
whether it is worth trading the next day. A ladder of rules and models then
predicts intraday direction, and every result is judged on returns after trading
costs, with uncertainty measured in days.

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

Each run answers the research questions from the same predictions:

| | Question | Where it shows up |
|---|---|---|
| Q1 | Does the gate pick better days? | `summary.md` gate table: the same predictions on approved vs. other days, with a 95% interval on the difference |
| Q2 | Do deep models beat simple rules and a linear model? | `summary.md` main table: `always_up`, `momentum_day`, `momentum_window`, `logreg`, then CNN1D, ResNet1D, InceptionTime |
| Q4 | Does an image view of the window help? | ResNet2D (Gramian Angular Field images) vs. ResNet1D, same stages and filters |
| Q5 | Does any edge survive costs? | net return at each cost level, and the breakeven cost |

Two further checks come free with the same predictions, no retraining:

- **Confidence filtering**: results using only the most confident predictions (`evaluation.coverage_levels`).
  Each fold's cutoff comes from earlier folds only, so nothing uses the future.
- **Within-ETF gate comparison**: the Q1 difference after subtracting each ETF's own average. The gate refuses
  whole ETFs that move only a few cents a bar, so the raw split partly compares expensive ETFs against cheap ones.

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
compared with the best fixed rule chosen from the past. It runs with and without the entropy gate's verdict
as input, which is the direct test of whether the entropy signal helps an agent that can learn from
feedback, and with two kinds of feedback:

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
| `src/screening.py` | Nightly gate readings and pass/fail rules |
| `src/data.py` | Alpaca download with caching, and the fixed 5-minute session grid |
| `src/features.py` | Causal channels, windowing, normalization, GAF/MTF encoders |
| `src/dataset.py` | Samples with gate verdicts attached; walk-forward splits |
| `src/baselines.py` | Rules that need no training |
| `src/models.py` | logreg / CNN1D / ResNet1D / InceptionTime / ResNet2D |
| `src/training.py` | Reproducible training with early stopping |
| `src/stats.py` | Trade scoring and day-block bootstrap intervals |
| `src/experiment.py` | Resumable sweep and the results report |
| `src/synthetic.py` | Bars with known structure for tests |
| `src/engine.py` | RL decision engine: policy, online walk-forward learning, baselines |
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

Built and tested: data, gate, features, models, sweep, report and decision engine (142 unit
tests plus the smoke test). Run on real data for SPY, the 12 ETFs and the S&P 500;
`etf_multihour` not yet. Findings are in [RESULTS.md](RESULTS.md). The live path (IBKR
execution, risk firewall) and position sizing are designed, not built.

Known limits: IEX volume is a sample of the consolidated tape; Kalman state
features are proposed, not built.
