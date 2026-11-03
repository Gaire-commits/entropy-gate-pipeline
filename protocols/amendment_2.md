# Amendment 2: how the listing exchange is chosen

Dated 2026-11-03, before the reserved period (2024-01-03 to 2026-10-12) was scored under either protocol, and before PE of daily returns had been compared with a variance baseline on any window. It is read together with `protocols/volatility_v2.yaml` and `protocols/portfolio_v2.yaml`, which do not change. **STEP 3 stays locked.**

## Why

The first real run of the data checks stopped, as the protocols require. Official opens covered 91.08% of usable bars in 2021, against the 95% limit. The limit stays at 95%.

The fault was in our code, not the data. Prints exist for 99.99% of 2021's bars on some exchange. `src/official.py` chose one exchange per stock per calendar year, and its note said Nasdaq's prints change code from T to Q at the start of 2021. They actually change during April to June 2021, flipping between the two. So Nasdaq stocks given Q for the year lost every print from January to March and some through June. Stocks that changed listing mid-year also lost prints, as HON's codes suggest it did.

## The fix, and how it was chosen

The comparison and its rule were committed in `4fc717f` before it ran (`scripts/official_listing_candidates.py`). A fix had to agree with the old rule's prices on at least 99.9% of the bars where both have an official open before the lock. Among those, the highest lowest-year coverage for 2016-2023 wins, with ties within 0.1 point going to the simplest.

| Rule | Lowest-year coverage, 2016-2023 | Agreement with the old rule | Bars differing by over 25 bp |
|---|---:|---:|---:|
| Old: per year | 91.08% | | |
| Per year, T and Q as one | 99.59% | 99.981% | 69 |
| Per month | 99.00% | 99.993% | 44 |
| **Per month, T and Q as one (the rule's pick)** | **99.78%** | 99.992% | 49 |

**The rule now.** For each stock and month, the listing exchange is the one with the most days whose closing print equals the vendor's close, with T and Q counted as one exchange. The official open and close come from that exchange, as before.

Coverage of usable bars by year, in percent. For the reserved years these are counts only:

| | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Old rule | 100.00 | 99.98 | 99.78 | 99.92 | 99.68 | 91.08 | 99.10 | 98.54 | 99.98 | 99.83 | 99.86 |
| New rule | 100.00 | 99.98 | 99.89 | 99.98 | 99.92 | 99.90 | 99.94 | 99.78 | 100.00 | 100.00 | 99.98 |

## What else the comparison showed

- **O and Q are the same price.** Official-open prints (Q) are rare before 2021, so earlier opens are mostly opening trades (O). On the 345,201 bars with both before the lock, they are equal within 0.5 bp in all of them, to four decimals. So the mix over time does not change the target.
- **2023-01-24, when the NYSE skipped most opening auctions.** Of 342 stocks then listed on the NYSE, under the old rule 265 had no official open and 77 had a Q print. Under A4's policy those 265 get no V that day, which suits a day when opening trades were unreliable. For the 77, the vendor's open reached 1,778 bp from the previous close and the official open 454 bp. This falls in validation, not the reserved period.

## Not settled

- **The high and low on 2023-01-24** may hold trades that were later cancelled, which would inflate the range for those 77 stocks. This was not checked. It is one day, in validation.
- **The 49 bars that differ by over 25 bp** were not examined one by one.
- **1,312 vendor bars before the lock still have no official open,** mostly VRT, CBOE, TPL and IBKR in years before they joined the index or moved listing, though that reason is inferred, not checked. Prints exist on another exchange for most, and are not used.
- **Numbers on the vendor's open do not change:** the baseline stand-in and the Clark-West calibration used no auction prints. The gate on the official open has still not run.

## What follows

STEP 1 is rerun under this rule. All its checks must pass before STEP 2, and the 95% limit and clustering rule stand unchanged. `tests/test_frozen.py` records the new hashes of `src/official.py` and `scripts/official_prices.py`.
