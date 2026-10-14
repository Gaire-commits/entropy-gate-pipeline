# Live plan (draft outline, 2026-10-14)

A plan to forward-test **the whole process** on **live companies, on days that have not happened yet**,
running by itself on AWS. Every trading day it fetches the bars, scores every arm, updates the paper books
and the report, and at the end date it writes the verdict. This is a draft. Nothing here is registered or
built.

## 0. What this is and is not

- **It is a forward test.** We freeze the rules today, then score every new trading day as it arrives. It is
  the one check the thesis is missing: a **second period**. E11 gave us a second sample of stocks, but over
  the same 2016–2026 years.
- **It is not real-money trading, and nothing places orders.** Every test so far says the edge is 0.1–0.6 bps
  gross against roughly 2 bps of cost, so a live book would lose money. The paper book below is reported
  only. If you ever want to trade for real, that is your decision and a separate project, and the data here
  argues against it.
- **Bonus: no survivorship bias.** Every historical test applied today's index lists back to 2016 (a disclosed
  weakness). Going forward, today's lists are the right lists.

## 1. What gets tested

| Arm | Question | Status before going live |
|---|---|---|
| **A. Own reversal** (primary) | A stock that lagged the market over the last 30 minutes recovers some of it over the next hour. | Established: E5, E10, E11. Frozen geometry, no new tuning. |
| **B. Correlated stocks** | When a stock's most-correlated peers move, does it follow within the next hour? | E5 found **sectors add nothing**. Data-driven peers have never been tested, so test them on 2016–2026 first (E12). It goes live only if it passes. |
| **C. Entropy and the reversal** | Is the reversal stronger when permutation entropy is low? | E6 null. Reported only, never judged. |
| **D. The original question** | Does the entropy gate plus a model trade profitably after costs? | Null on IEX. The SIP sweep (section 22) must be validly redone first. The model is trained once on 2016–2026 and frozen. Reported, with its own registered rule. |

**Arm A, fixed from the history (nothing chosen after seeing new data):**
- **Universe:** the S&P 500 and the S&P MidCap 400 as of the start date, about 900 stocks.
- **Signal:** each stock's market-relative return over the last 6 five-minute bars, sign flipped.
- **Outcome:** the relative return over the next 12 bars (1 hour), entering 1 bar later. The 30-minute hold is
  also reported, as the E5 geometry.
- **Measure:** cross-sectional rank IC per moment, averaged per day, with day-block intervals. The same code
  path as E10/E11 (`src/horizons.py`, `src/reversal.py`).

**Arm B, the correlated-stocks idea, to be registered as E12 on 2016–2026 first:**
- **Peers:** at the start of each month, each stock's k = 5 stocks whose 5-minute relative returns correlated
  most over the previous 60 trading days (causal: past data only).
- **Signal:** the peers' average relative move over the last 30 minutes.
- **Outcome:** the stock's own relative return over the next hour.
- **Control:** the same test with k random peers (as E5's shuffled sectors), on the same moments. The own
  reversal is in both, so it cancels out.
- **Rule:** real peers minus random peers > 0 at 99%. Low prior: E5's sector version found nothing.
- **Reported:** whether the stocks that move first (the "leaders") have any pull on the rest.

## 2. Running it: one daily job on AWS

The rule is frozen before the data exists. That is what makes a forward test honest, not watching it in real
time. A job after each close is enough, and a missed day is filled in on the next run (Alpaca keeps the
SIP bars).

| Piece | AWS service | Job |
|---|---|---|
| Timer | EventBridge Scheduler | Runs at 16:30 ET each weekday (the job skips market holidays itself) |
| Job | Lambda from a container image (Fargate if it outgrows 15 minutes) | Fetch, score every arm, paper books, ledger, report |
| Storage | S3 bucket | `bars_sip_forward/` (its own folder, never mixed with history), `ledger.csv`, `report.md`, frozen models |
| Keys | Secrets Manager | Alpaca keys. Never in code, logs or chat |
| Alerts | SNS email | On a failed run, a coverage-guard refusal or 3 missed days; a weekly summary |
| Spend | AWS Budgets | Alarm at $10 a month. Expected about $1–5 |

**Each run, in order:**
1. **Catch up:** find every trading day since the last stored one, and fetch those days' 5-minute SIP bars.
2. **Guard:** refuse a day under 90% coverage (alert, retry next run), and never mix feeds.
3. **Score:** arms A–D on the new days, with the frozen code version and config hash written into every
   ledger row.
4. **Paper books:** append each day's gross and net.
5. **Report:** rewrite `report.md` (cumulative IC against the historical band, coverage, paper books).
6. **Verdict:** on the registered end date, apply the rules and write `verdict.md`. Nobody chooses when to
   stop.

**Reproducible:** the job, its tests and the deploy template (AWS SAM) live in the repo under `deploy/`. The
image is tagged with the commit it was built from. The same code can rescore the whole ledger from the
stored bars, so the AWS run can be checked against a Colab rerun.

**Reading the results:** a read-only AWS user, so Claude can read `report.md` and the ledger. Or the job
also copies the report to Drive.

**Optional, reported only:** an IBKR or IEX real-time capture for a live view of the signal during the day
(`src/ibkr_capture.py` exists, market data only). It needs an always-on machine and never feeds the verdict.

## 3. Paper book (reported, never judges)

- At each signal time: long the 10% of stocks that lagged most, short the 10% that led most, equal weight,
  held 1 hour. Purely on paper, computed from bars, with no orders anywhere.
- Report gross, net at 2 bps per side, and net at each stock's measured quoted spread (from IBKR quotes if
  the capture runs).
- Expected: positive gross, negative net. Showing that in live data is itself a result.

## 4. Pre-registration (`configs/sip_e13_forward.yaml`, frozen by hash before day 1)

- **Start date and fixed length.** No stopping early because it looks good or bad. Weekly looks are reported,
  and the verdict is given once, at the end.
- **Length from a power calculation, not a guess.** Rough estimate from E5 (t 7.5 over about 2,700 days):
  about 130–190 trading days (6–9 months) to see the reversal at the strength we measured. Step 1 below
  computes it properly from the saved per-day ICs.
- **Verdict (Arm A):**
  - **Holds forward:** mean daily IC > 0 at 95%, one-sided, and the point estimate at least half the
    historical one.
  - **Weaker forward:** IC > 0 but below half.
  - **Not seen:** neither.
- **Verdict (Arm B):** the E12 rule, applied to forward days, and only if E12 passed.

## 5. Monitoring

The job rewrites `report.md` daily and emails a weekly summary:
- Cumulative mean IC against the band the history predicts.
- Days covered and days missing.
- The paper books, gross and net.
- Peer-arm IC against the random-peer control.

## 6. Build order

1. **Power calculation** from the saved E10 and E11 summaries → the registered length. Small.
2. **E12, correlated peers on 2016–2026:** `src/peers.py`, tests with planted followers and a shuffled control
   first, then a config, then a Colab section. About a day of building and one Colab run.
3. **Redo section 22** (the SIP sweep) validly, if arm D is kept. Hours of Colab.
4. **The forward job:** catch-up fetch, scorer, paper books, report, verdict, with tests on synthetic days
   (including missed days and half days). Runs locally first, then in a container.
5. **AWS (your part):** create the account, install the AWS CLI and Docker, run `aws configure`, and put the
   Alpaca keys in Secrets Manager. Claude never handles keys.
6. **Register E13** (forward) with the start date and freeze it. Then deploy, which you approve, since it
   creates billable resources.
7. **Verdict** at the end date, written by the job and recorded like every other test.

## 7. Risks and honest notes

- **The likely result** is that the reversal persists and the paper book loses after costs. That is fine for the
  thesis: an out-of-sample-in-time replication is the strongest evidence it can have.
- **Regimes change.** If the reversal weakens, that is a finding, not a failure. The verdict has a "weaker
  forward" level for exactly this.
- **Arm B is a long shot** after E5, which is why it has to earn its place on 10 years of history first.
- **Alpaca's free plan** gives SIP only after a 15-minute delay, which is fine, since nothing here is real-time.
  Real-time SIP would cost about $99 a month and is not needed.
- **Silent failure** is the main risk of an unattended job: alerts on every failure and on missed days, and a
  catch-up step so a gap costs nothing.
- **The index lists change.** The universe is frozen at the start date. Stocks that leave an index stay in;
  newcomers are not added. Delistings are logged.
- **Thesis timing.** Six to nine months of forward data may outrun the writing schedule. The write-up can
  start now, with the forward test as a final section or a dated addendum.
