# Handover

Written 2026-10-08. A snapshot to restart work from, for Kshitiz or for a new Claude session.
[RESULTS.md](RESULTS.md) is the full record of results and [README.md](README.md) explains how
everything works; this file is the short version, plus what is unfinished.

## The project in one paragraph

A thesis pipeline (Kshitiz Gaire, Fordham). It asks whether permutation entropy can pick out predictable
periods in 5-minute price bars well enough to trade them after costs. Data: Alpaca's free IEX 5-minute bars
from July 2020. It tests a model ladder (rules, logistic regression, gradient-boosted trees, CNN1D, ResNet1D,
InceptionTime, ResNet2D, plus versions with entropy inputs), using walk-forward validation (15 quarterly
folds, 940+ test days) and day-block bootstrap intervals. On top sit an RL decision engine and an online RL
agent. Universes: SPY, 12 sector ETFs, the S&P 500, 37 cross-asset ETFs, and a Nasdaq-100 hold-out.

## Where everything is

| What | Where |
|---|---|
| Code | `~/entropy-gate-pipeline`, GitHub `Gaire-commits/entropy-gate-pipeline` (private), branch `main` |
| Tests | 287 pass (`python3 -m pytest -q`, about 2 min). `tests/test_frozen.py` fails if a pre-registered config is edited |
| Bars and outputs | Google Drive: `My Drive/entropy-gate-pipeline-data/{data,outputs}`. Not on this Mac and never committed |
| Runner | `notebooks/colab.ipynb` on Colab Pro. Cells 1–3 mount Drive, pull the code and link `data/` and `outputs/` to Drive |
| Reading results | Kshitiz runs a Colab section; Claude reads `outputs/<experiment>/*.md` from Drive through the Google Drive connector (`search_files`, then `read_file_content`) |

**Colab runtime:** use CPU with High-RAM for anything on the S&P 500 (trees, engine, breadth, online agent).
Use a GPU only for the deep models.

**Notebook sections:**

| Section | What it runs |
|---|---|
| 4 | Checks |
| 5 | Download bars |
| 6–8 | SPY and ETFs |
| 10 | S&P 500 |
| 11 | Decision engine |
| 12 | Entropy as features |
| 13 | Selective trading |
| 14 | Breadth |
| 15 | Cross-asset |
| 16 | Nasdaq-100 hold-out |
| 17 | Skill checks and replication |
| 18 | Online agent |
| 19 | E1: the volatile-moment lead |
| 20 | E2: model or signal |
| 21 | SIP feed check and fetch |
| 22 | SIP reruns with the existing rules |
| 9 | All results |

## Where things stand

**The latest run is not recorded yet.** Section 18 (the online agent) finished on 2026-10-07. Its result is
below and on Drive, but **not yet in RESULTS.md**. Everything before it is recorded and pushed.

### Findings so far

1. **Nothing beats trading costs.** This holds on SPY, the 12 sector ETFs at two horizons, the S&P 500 and
   37 cross-asset ETFs, for every model and rule at a 1–2 bps round trip. The best gross return is about
   +1 bps per trade.
2. **Model complexity buys nothing.** No deep model is the best row anywhere. Trees are not clearly better
   than logistic regression.
3. **Entropy adds nothing in any of three roles:**
   - As a gate: 7 of 93 comparisons are "significant", about the 5 chance would give, and 4 of the 7 point the
     wrong way.
   - As model inputs: −0.45 to +0.27 bps, with no consistent sign.
   - As engine inputs: more trading and bigger losses.

   A likely reason: ordinal patterns of 5-minute returns are nearly blind to slow drift.
4. **The "trade only confident signals" lead was volatility.** Picking the most volatile moments, with no
   model at all, does as well. A side finding remains unchecked: +15.21 bps gross on the S&P 500's most
   volatile 2% of moments, mostly long.
5. **Diversification was an illusion.** 768 S&P 500 trades a day were 7–14 independent bets. With the market
   removed, direction-trained models have a cross-sectional IC of about 0. The data could have detected an IC
   of 0.0075, and a profitable book needs 0.0154.
6. **The Nasdaq-100 hold-out was a null by its pre-registered rule.** A tree model trained to beat peers on
   418 S&P 500 stocks ranked the 85 unseen Nasdaq-100 stocks above chance (IC +0.0141, t 3.37). Its book made
   +1.47 bps gross and −0.53 net.
7. **That skill did not replicate.** On the 416 training stocks, scored out of time, the IC was +0.0005
   (t 0.2) and the timing IC −0.0055. On the 85, the IC is mostly a fixed preference for certain stocks;
   timing is +0.0019 (t 0.4). Survivorship in today's index lists can produce exactly that.
8. **The RL decision engine finds nothing.** It stays flat on the ETFs. On the raw S&P 500 it first collapsed
   into always-long; that was fixed with market-neutral rewards, after which it also stays flat.
9. **The online agent (section 18) is a null.** Details below; this result is unrecorded.

### Online agent result (Drive: `outputs/online_sp500/online.md` and `online_nasdaq100_in_sp500.md`)

4,631 signal moments from 2020-07-28 to 2026-10-06. The replay check on 40 real stocks passed: all 106,210
batch samples were reproduced exactly (window difference 5.1e-08, return difference 0).

Net bps per opportunity, flat counted as zero, market-neutral, 2 bps per position:

| Run | Greedy net [95% CI] | Trades | Explore | Frozen | Placebo mean (runs) | p |
|---|---|---:|---:|---:|---|---:|
| **full + entropy (main)** | **−0.73** [−1.14, −0.35] | 42% | −0.84 | −0.69 | −0.31 (19) | 1.00 |
| bandit + entropy | −0.58 [−0.92, −0.25] | 44% | −0.74 | −0.99 | −0.78 (4) | 0.20 |
| bandit, no entropy | −0.32 [−0.59, −0.06] | 24% | −0.34 | −0.99 | −0.54 (4) | 0.20 |

- **Exploration and drift.** Mean entropy was 0.50 / 0.50 / 0.15 nats (target 0.5). The drift detector
  never fired in any run.
- **The 85 Nasdaq-100 stocks** (scored only; the agent learned from all stocks):
  - Main run: −0.51 [−1.10, +0.07].
  - Bandit without entropy: +0.08 [−0.47, +0.64]. That is not a finding: 4 placebos and many looks.
  - The main run's greedy book was negative every year, worst in 2023 (−1.17).
- **Reading.** A null by the protocol in `configs/online_sp500.yaml`. Gross is about +0.1 bps per
  opportunity, against about 0.8 bps of cost at a 42% trade rate.
- **Real lost more than all 19 placebos.** A guess at why: a tiny real edge keeps the agent trading, while in
  the placebo it may learn to trade less. **Not checked**: the report does not show the placebos' trade rate.
- **Exploration only pays when there is a signal to rediscover.** In the synthetic test where a signal flips
  sign, entropy control recovers it (+1.6 to +3.0 bps) and no entropy fails (−3.4 to −0.03). On real data,
  where there is no signal, the agent with no entropy bonus lost least because it traded least.

## Unfinished: do these next

0. **Switch to the SIP feed, then rerun.** Decided 2026-10-09, before any SIP result: SIP (consolidated)
   is the headline dataset, and IEX results become the appendix. See RESULTS.md, "Data: SIP becomes the
   headline feed".
   - **Section 21:** `feed_check.py` first. Read `outputs/feed_check.md` to confirm the account serves SIP
     from 2016. Then the fetch into `data/bars_sip`, about 1–2 hours.
   - **Section 22:** screen and sweep, E1 (untouched period now 2016 to July 2020), E2, hold-out and
     replication. All use `configs/sip_*.yaml` with the existing rules.
   - **Never point a SIP config at `data/bars`.** The fetch refuses it (`FeedMismatch`), because it would
     overwrite the IEX bars behind every earlier result.
   - **E1 and E2 on IEX ran on 2026-10-08 but aren't recorded in RESULTS.md yet:**
     - E1: not replicated. The rule without a model: +7.70 [−3.03, +18.17] in the test quarters; 57% of the
       gain came from Mar–May 2025; the stress cost averaged about 8 bps.
     - E2a: signal-limited. No cell reached t ≥ 3; the default reproduced the replication exactly.
     - E2b: momentum adequate, interaction not adequate. The detection limit for simple edges is about
       IC 0.015.
   - **The agreed plan:**
     1. SIP reruns.
     2. A better formulation: longer horizons, residual or market-neutral targets.
     3. A few economically motivated features, built as explicit interactions, each checked with planted
        edges.
     4. Serious tuning only after a signal reproduces.

1. **Record section 18 in RESULTS.md and push.**
   - Add an "In short" bullet.
   - Replace "results pending" in the "Online agent on streaming bars" section with the table above.
   - Add a progress-log row.
   - Update next step 2.
2. **Add the placebo runs' trade rate to `scripts/online_replay.py`'s report.** This checks the explanation for
   "lost more than the placebo".
3. **Find the 2 missing stocks.** The replication scored 416 of the 418 training stocks; which two are missing
   is unknown.
4. **Leave one date label as it is.** `configs/ndx_holdout.yaml` was right; `configs/ndx_replication.yaml` says
   its rule was fixed 2026-10-08, but the commit (`079f520`) shows 2026-10-07 17:55, before the results.
   RESULTS.md already notes this. Don't edit the pre-registered configs.
5. **Then the open list in RESULTS.md "Next steps":**
   - Check the volatile-moment side finding.
   - Run `etf_multihour`.
   - Optionally measure real costs with the IBKR recorder.
   - Fix the "ETF" wording in the S&P 500 gate table.
   - Count gate approvals as episodes, not days.
   - Use point-in-time S&P 500 membership.
   - Write up.

   The direction agreed so far is to stop adding complexity soon and write up a careful null, together with its
   methodological findings.
6. **A live paper feed was discussed but not built.** It would feed Alpaca websocket bars into the same
   `StreamFeaturizer` and `OnlineAgent`. Only worth it if something survives.

## How work is done here

- **Pre-register.** Every new test writes its success rule into the config, with a date, before running.
  Results are judged only by that rule: no promoting another model, cost or book size afterwards.
- **Controls first.** Every method gets synthetic positive and negative controls in `tests/` before real data.
  The test names state the guarantee.
- **Inference by day.** Intervals are day-block bootstraps. Cross-sectional work is market-neutral.
- **Recording loop:**
  1. Kshitiz runs a Colab section and says "check now".
  2. Claude reads the reports from Drive and explains them plainly.
  3. On "record", Claude updates RESULTS.md (and README if needed) and commits.
  4. Claude pushes to `main`. Never force-push.
- **Commit attribution:** commits end with the `Co-Authored-By` line the session asks for.
- **Communication:** Kshitiz prefers short, plain-language summaries with the key numbers. "Build on it"
  means go ahead and implement.

## Hard rules

- **Never handle credentials in chat.** Alpaca keys and the GitHub token live only in the gitignored `.env`
  or in Colab Secrets. Never print or echo them.
- **Nothing places orders.** The IBKR tooling is market-data only, and every position here is a paper
  position.
- **Keep data out of git.** Data and outputs stay on Drive.
- **Report what happened.** Every list is today's constituents, so survivorship bias must be disclosed with
  results. Nulls are reported as nulls.

## Map of the code

| File | Role |
|---|---|
| `src/data.py` | Alpaca download, cache, fixed 5-minute session grid |
| `src/features.py` | Causal windows and channels |
| `src/dataset.py` | Builds the samples |
| `src/entropy.py` | Permutation entropy and the shuffle test |
| `src/screening.py` | Nightly gate |
| `src/entropy_features.py` | Continuous entropy inputs |
| `src/models.py` | Neural models |
| `src/ml.py` | Trees |
| `src/training.py` | Training loop |
| `src/experiment.py` | Walk-forward sweep, predictions, report |
| `src/selective.py` | Confidence filtering and volatility placebo |
| `src/crosssection.py` | Breadth, IC, long-short books, skill decomposition |
| `src/holdout.py` | Train/trade universes, relative target |
| `src/engine.py` | Fold-by-fold RL engine (bandit and full feedback, market-neutral) |
| `src/online.py` | Streaming featurizer, online agent, entropy control, Page-Hinkley drift detector, placebo replay |
| `src/ibkr.py`, `src/ibkr_capture.py` | Local market-data recording |

Scripts: `sweep.py`, `summarize.py`, `engine.py`, `selective.py`, `breadth.py`, `skill_checks.py`,
`online_replay.py`, `build_universe.py`, `fetch_data.py`, `run_screen.py`, `smoke_test.py`. Each script's
docstring gives its usage.

Configs: `spy_gao`, `etf_gao`, `etf_intraday`, `etf_multihour` (not yet run), `sp500_multihour`,
`cross_asset_multihour`, `ndx_holdout`, `ndx_replication`, `online_sp500`.
