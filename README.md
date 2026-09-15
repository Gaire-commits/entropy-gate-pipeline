# Entropy Gate Pipeline

Predictability screening before prediction. An ordinal-pattern entropy gate decides
which symbols a model is allowed to trade, so no forecast is forced onto a process
with no exploitable structure.

## Setup

**Running on Google Colab (GPU training):** open [`notebooks/colab.ipynb`](notebooks/colab.ipynb) in
Colab. It clones this repo, mounts Drive so `data/` and `outputs/` survive between sessions, and reads
credentials from Colab's Secrets panel — nothing is pasted into a cell in plain text.

**Running locally:**

```bash
pip install -r requirements.txt
cp .env.example .env     # then paste your Alpaca keys into .env
```

`.env` is gitignored. Credentials are read from the environment only — no key is
ever passed as an argument or written into a config.

Verify the install without credentials:

```bash
python scripts/smoke_test.py
python -m pytest tests/ -q
```

## Pipeline

```
fetch_data.py  ->  run_screen.py  ->  train.py
   Alpaca            entropy gate      walk-forward CV
   -> parquet        -> screen.parquet -> metrics + summary
```

```bash
python scripts/fetch_data.py       # cache bars into data/bars/
python scripts/run_screen.py       # entropy readings + capacity diagnostic
python scripts/train.py --arch resnet1d --gate
```

Everything is driven by `config.yaml`; experiments should differ by config, not by
edited code.

## Experiments

| Question | Command |
|---|---|
| Does the gate help? | `train.py --arch resnet1d --gate` vs `--no-gate` |
| Does depth beat a linear model? | `train.py --arch logreg` vs `--arch inceptiontime` |
| Does the image framing beat the sequence framing? | `train.py --arch resnet2d` with `encoding: gaf` vs `--arch resnet1d` with `encoding: 1d` |
| How much cost can it absorb? | `train.py --cost-bps 5` vs `--cost-bps 15` |

## Layout

| Path | Role |
|---|---|
| `src/entropy.py` | Permutation entropy, weighted PE, complexity-entropy plane |
| `src/screening.py` | Per-session gate + capacity diagnostic |
| `src/features.py` | Causal windowing, GAF/MTF encoders |
| `src/models.py` | logreg / CNN1D / ResNet1D / InceptionTime / ResNet2D |
| `src/dataset.py` | Dataset assembly, walk-forward splits |
| `src/training.py` | Training loop, cost-adjusted evaluation |

## Design notes

**The gate never sees forward returns.** Selection uses ordinal patterns of past
returns only, so it cannot leak label information into which symbols get traded.

**A gate reading applies to the next session.** Entropy is computed after each close
and decides eligibility for the following day (`trade_date`), which is how it runs
live as a nightly batch. Matching a reading to the day it was computed would let a
morning window be admitted by entropy that already saw that afternoon.
`tests/test_screening.py` pins this down.

**Ties are a trap.** A series of repeated quotes reads as highly ordered — the smoke
test's stale fixture scores PE 0.11, comparable to a sine wave. `tie_fraction` is
reported alongside every reading and `apply_gate` refuses names above the threshold.

**Intraday means intraday.** `window + embargo + horizon` must fit inside one
session or `make_windows` raises. Checking the feature window and the trade
separately is not enough — that admits samples whose features come from one day and
whose fill comes from the next. Overnight holding is expressible, but has to be
opted into with `allow_overnight`.

**Costs are in the evaluation, not bolted on after.** Every result reports gross and
net basis points per trade plus the breakeven cost level, because a model can be
reliably right about direction and still lose money once the spread is paid.

**The positive control matters.** `smoke_test.py` plants a horizon-matched signal and
checks the trainer recovers it (~0.72 accuracy). Without that, a null result on real
data is indistinguishable from a broken training loop.

**ResNet2D is a structural twin of ResNet1D** — same three residual stages, same
64/128/128 progression — so the GAF comparison isolates the sequence-vs-image framing
rather than confounding it with depth or capacity. It is the honest way to ask whether
an image architecture belongs on market data at all: GAF genuinely converts temporal
correlation into 2-D spatial structure, which is the only thing that would justify it.
Mismatching an architecture with an encoding raises immediately instead of failing
inside a convolution.

## Status

Screening, features, models, dataset assembly, and walk-forward training are
implemented and tested (25 tests). The live path — real-time gate, risk firewall,
IBKR execution — is designed but not built.
