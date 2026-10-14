# Handover

Updated 2026-10-14 (after E12). A snapshot to restart work from, for Kshitiz or for a new Claude session.
[RESULTS.md](RESULTS.md) is the full record of results and [README.md](README.md) explains how
everything works; this file is the short version, plus what is unfinished.

## The project in one paragraph

A thesis pipeline (Kshitiz Gaire, Fordham). It began by asking whether permutation entropy can pick out
predictable periods in 5-minute price bars well enough to trade them after costs. It tested a model ladder
(rules, logistic regression, gradient-boosted trees, four deep architectures, versions with entropy
inputs), an RL decision engine and an online RL agent, on SPY, sector ETFs, the S&P 500, cross-asset ETFs
and a Nasdaq-100 hold-out. Nothing beat costs and entropy added nothing. On 2026-10-13 a control inside the
lead-lag test turned up the first real pattern, a short-term reversal. The research now builds on that
pattern and is judged on whether structure exists and when it is stronger, not on whether it pays.

## Where everything is

| What | Where |
|---|---|
| Code | `~/entropy-gate-pipeline`, GitHub `Gaire-commits/entropy-gate-pipeline` (private), branch `main` |
| Tests | 348 pass (`python3 -m pytest -q`, about 3 min). `tests/test_frozen.py` fails if a pre-registered config is edited |
| Bars | Google Drive, `My Drive/entropy-gate-pipeline-data/data`. `bars/` is IEX (from July 2020); `bars_sip/` is SIP (503 stocks, 2016-01-04 to 2026-10-12, 98–99.7% of slots filled). Never committed |
| Outputs | Drive, `.../outputs/<experiment>/`. Claude reads the `.md` reports through the Google Drive connector |
| Runner | `notebooks/colab.ipynb` on Colab Pro. Sections 1–3 mount Drive, pull the code and link `data/` and `outputs/` |

**Colab runtime:** CPU with High-RAM for everything on the S&P 500. A GPU only for the deep models.

**Data feed:** SIP is the headline dataset since 2026-10-09; IEX results are the robustness appendix. The two
feeds must never share a cache folder, and the fetch refuses to overwrite one with the other.

**Notebook sections:**

| Section | What it runs |
|---|---|
| 4–8 | Checks, download, SPY and ETFs (IEX) |
| 10–15 | S&P 500, engine, entropy features, selective, breadth, cross-asset (IEX) |
| 16–17 | Nasdaq-100 hold-out, skill checks, replication (IEX) |
| 18 | Online agent (IEX) |
| 19–20 | E1 volatile-moment lead, E2 model or signal (IEX) |
| 21 | SIP feed check and fetch (done) |
| 22 | SIP reruns of the sweep, E1, E2, hold-out, replication (**not validly done**, see below) |
| 23 | E5 lead-lag |
| 24 | E6 entropy and the reversal |
| 25 | E7 the reversal as a slope |
| 26 | E8 the sharper slope |
| 27 | E9 horizons from 30 minutes to 2 hours |
| 28 | E10 how long the reversal lasts |
| 29 | E11 the reversal in mid caps |
| 30 | Forward-test length (a design calculation, not a test) |
| 31 | E12 correlated peers |
| 9 | All results (old summaries only) |

Do not use "Run all": it reruns every old section.

## Findings

### The main positive result (SIP, 2016–2026, 503 stocks, 13.9 million stock-moments)

1. **A real short-term reversal (E5).** A stock that lagged the market over the last 30 minutes recovers a
   little over the next 30: cross-sectional IC +0.0064 [+0.0047, +0.0080], t = 7.5, positive in both halves
   and in 8 of 11 years. Tiny in money terms: +0.12 bps gross per position.
2. **A stock's sector adds nothing to it (E5).** The registered lead-lag signal worked no better with real
   sectors than with shuffled ones (difference −0.0008 [−0.0016, −0.0000]). Verdict: own reversal, not
   lead-lag.
3. **Permutation entropy does not say when it is stronger (E6).** The reversal's IC is the same in every third
   of entropy: +0.0001 [−0.0018, +0.0020] at 99%, and zero among stocks of similar volatility. The cleanest
   test the gate idea has had, and a null. Volatility leans stronger (t 3.1), reported only.
4. **As an amount it is tiny, and it is not confined to large moves (E7, E8).** E7 passed narrowly (largest
   minus smallest third of moves, +1.12% [+0.05%, +2.49%] of the lag recovered). E8, the same question with an
   instrument twice as sharp, did not confirm it: +0.07% [−0.55%, +0.67%]. Over all stock-moments the share
   recovered is +0.28% [−0.06%, +0.64%] at 99%. So: a reliable ordering, worth under a tenth of a basis point.
   The rank correlation is about twice what the slope implies; why is not established.
5. **The reversal completes within about an hour and lasts through 4 hours (E9, E10).** On the same moments
   the IC is +0.0063 at 30 minutes, +0.0098 at 1 hour and about +0.0100 at 2 and 4 hours; the 4 hours vs 30
   minutes difference is +0.0038 [+0.0001, +0.0076] (a hair), all of it in the first hour. A full trading
   day holds less (+0.0053, gross −0.24 bps): an intraday effect. E9's best cell (+0.0151, +0.77 bps) was
   flattered by selection; on comparable moments it is +0.0101, +0.33 bps. The 10:00 signal is strongest.
6. **It replicates on a second sample of stocks (E11).** On 399 S&P MidCap 400 stocks the project had never
   used, all three registered claims hold: it exists (+0.0051, t 4.0), builds over the first hour (+0.0029
   [+0.0007, +0.0049]) and does not carry overnight (−0.0054 [−0.0088, −0.0021]). Same strength as in large caps
   (−0.0012 [−0.0035, +0.0011]): the prediction that less liquid stocks reverse more is not borne out.
7. **Correlated stocks do not lead (E12).** Each stock's 5 most correlated stocks (previous 60 days, monthly), with
   the stock's own move removed, predict no better than random peers: +0.0012 [−0.0009, +0.0034] at 99%. 83% of
   the chosen peers are in the stock's own sector. The correlated-peers arm is dropped from the forward test.
8. **No alpha.** Nothing earns a return after costs; the largest amount (+0.41 bps, 4 hours) is about 20% of a
   2 bps cost.

### Earlier results (IEX unless noted)

9. **Nothing beats trading costs** on SPY, the ETFs, the S&P 500 or cross-asset, for any model or rule.
10. **Model complexity buys nothing.** Deep models are no better than logistic regression or trees.
11. **Entropy adds nothing** as a gate, as model inputs or as engine inputs.
12. **The "confident trades" lead was volatility.**
13. **768 trades a day were 7–14 independent bets;** with the market removed, direction-trained models have no
   stock-level skill.
14. **The Nasdaq-100 hold-out skill did not replicate** (IC +0.0141 on 85 stocks; +0.0005 on 416 others).
15. **The RL engine and the online agent find nothing.** The online agent lost more than all 19 placebos.
16. **The volatile-moment lead is not replicated (E1), on IEX and on SIP.** 57–61% of it came from
    Mar–May 2025; volatility-scaled costs would exceed it.
17. **Model or signal (E2):** signal-limited for simple edges; the trees recover only a third of a
    conditional edge. Detection limit for a simple edge about IC 0.015.

## Unfinished

1. **The forward test ([LIVE_PLAN.md](LIVE_PLAN.md)).** Freeze the rules, then score every new trading day by
   a daily AWS job (no orders). Done: the plan, E12 (the correlated-peers arm is dropped). Next: section 30's
   length calculation (its first Colab run wrote nothing; the script passes its end-to-end test), then the
   forward job, then the AWS setup (Kshitiz's account and keys), then registering E13 and deploying.
   Write-up can start in parallel.
2. **E7's "largest tenth" row was wrong in its first report** (a script bug, fixed and tested). Rerun
   `scripts/reversal_slope.py --config configs/sip_e7_slope.yaml` before quoting that row. The verdict never
   used it.
3. **Section 22 (SIP reruns of the sweep, E2, hold-out, replication) has no valid result.** It first ran on an
   unfinished fetch (56 of 503 stocks), and a later rerun reused those files. To redo it: run the set-aside
   cell at the top of section 22 (moves the old files, deletes nothing), then the cells. The screen needs
   `--workers 0`. Hours. E1 on SIP is valid and recorded.
4. **Possible next tests on the reversal,** each to be registered before running:
   - Which large moves reverse: trading volume (heavy-volume moves tend to continue, thin-volume moves to
     reverse; Campbell, Grossman and Wang 1993; Llorente and co-authors 2002).
   - A second sample: a different universe or period, since E6–E8 all use the same data.
5. **Parked, on disk, not committed:** `src/folding.py` and `tests/test_folding.py`, the time-of-day
   ("pulsar folding") test. Passing, no config, never run.
6. **Proposed, not built:** thesis figures (the IC summary chart, a test-registry timeline, the planted-edge
   curve); a cross-stock correlation-entropy figure; a correlation-weighted lead-lag (pointless now that
   sectors add nothing).
7. **Small leftovers:** the placebo runs' trade rate in the online report; which 2 of the 418 training stocks
   were not scored; the "ETF" wording in the S&P 500 gate table.

## How work is done here

- **Pre-register.** Every test writes its rule into its config, with a date, before running. Results are
  judged only by that rule. Configs are frozen by hash; a new rule gets a new config.
- **Controls first.** Every method gets synthetic positive and negative controls in `tests/` before real data.
- **Check the instrument before trusting it.** Twice a first idea failed on synthetic data and was dropped
  (the "sign" slope; a too-narrow episode definition).
- **Inference by day.** Intervals are day-block bootstraps. Cross-sectional work is market-neutral.
- **Yardstick since 2026-10-13:** does structure exist, and when is it stronger. Costs are reported and
  decide no verdict.
- **Recording loop:**
  1. Kshitiz runs a Colab section and says "check now".
  2. Claude reads the report from Drive and explains it plainly.
  3. Claude records it in RESULTS.md, commits and pushes. Never force-push.
- **Void runs are disclosed,** not hidden (see RESULTS.md, E5 section).
- **Communication:** short, plain-language summaries with the key numbers. "Build on it" means implement.

## Hard rules

- **Never handle credentials in chat.** Alpaca keys and the GitHub token live only in the gitignored `.env`
  or in Colab Secrets.
- **Nothing places orders.** The IBKR tooling is market-data only; every position here is a paper position.
- **Keep data out of git.**
- **Never edit a pre-registered config.**
- **Scripts refuse a cache under 90% complete.** Do not work around it.
- **Survivorship bias is disclosed:** every list is today's constituents, applied back to 2016 on SIP.

## Map of the code

| File | Role |
|---|---|
| `src/data.py` | Alpaca download, cache (one feed per folder), fixed 5-minute session grid |
| `src/features.py`, `src/dataset.py` | Causal windows, channels, samples |
| `src/entropy.py`, `src/screening.py`, `src/entropy_features.py` | Permutation entropy, the gate, entropy inputs |
| `src/models.py`, `src/ml.py`, `src/training.py`, `src/experiment.py` | Models, trees, training, walk-forward sweep |
| `src/selective.py`, `src/crosssection.py`, `src/holdout.py` | Confidence filtering, breadth and IC, hold-out |
| `src/engine.py`, `src/online.py` | RL engine; streaming featurizer and online agent |
| `src/volatility_lead.py`, `src/capacity.py` | E1; E2 (learning curve, capacity, planted edges) |
| `src/feed_compare.py` | IEX against SIP |
| `src/leadlag.py` | E5: sector lead-lag with a shuffled-sector control |
| `src/reversal.py` | E6–E8: the reversal by state, and as a slope |
| `src/ibkr.py`, `src/ibkr_capture.py` | Local market-data recording |

Each script's docstring gives its usage. Experiments on the reversal: `scripts/leadlag.py`,
`scripts/reversal.py`, `scripts/reversal_slope.py`.
