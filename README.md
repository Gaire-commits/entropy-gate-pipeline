# Entropy Gate Pipeline

Predictability screening before prediction, on SPY and the 11 Select Sector SPDR ETFs.
A nightly gate reads the ordinal structure of each ETF's recent returns and decides
whether it is worth trading the next day. A ladder of rules and models then
predicts intraday direction, and every result is judged on returns after trading
costs, with uncertainty measured in days.

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

python scripts/fetch_data.py         # SPY + 11 sector ETFs, 5-min bars from 2016
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

**Time is fixed to the clock.** IEX only prints a bar when a trade happens on IEX.
Missing bars are restored as gaps on a fixed 5-minute grid, so "the last half-hour"
always means 3:30–4:00 and a gap drops the affected window instead of shifting it.

**Intraday means intraday.** A window, its execution lag and its hold must fit in
one session, or sample construction raises.

**Positive controls.** `smoke_test.py` plants signals in both setups and checks that
the rules and models recover them. Without that, a null result on real data is
indistinguishable from a broken pipeline.

## Status

Built and tested: data, gate, features, models, sweep, and report (79 unit tests plus
the smoke test). Not yet run on real data in this form. The live path (IBKR
execution, risk firewall) and the RL sizing layer are designed, not built.

Known limits: IEX volume is a sample of the consolidated tape; Kalman state
features are proposed, not built.
