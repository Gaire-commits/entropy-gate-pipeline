# Amendment 4: the volatility test from daily bars only

Dated 2026-11-17. **This comes after the portfolio protocol was scored on the reserved period, and before the volatility protocol was.** PE of daily returns had still not been set against variance on any window. It is read together with `protocols/volatility_v2.yaml` and amendments 1 to 3.

## Why

The question is about next-day volatility, from daily data: V comes from each day's open, high, low and close, and the HAR inputs and PE come from daily bars. The only role of the 5-minute cache in the primary test was a cross-check. A day was dropped when its daily high-low range and the 5-minute bars' range disagreed by more than 25%. Through 2024-01-02, this dropped 525 of roughly 950,000 stock-days.

Kshitiz asked that the test not depend on 5-minute bars. The immediate cause was that STEP 3 failed before loading anything, because that cache was not found on Drive.

## What changes

- **The range cross-check is dropped.** Every other rule stays as it was: missing values, truncated days (found from the vendor's daily trade counts), corporate actions, identity breaks, the official open, eligibility, the baseline rule, the gate, the test, the critical value and the verdict. `src/volpanel.py` and `src/volforecast.py` run without the sessions file when none is given.
- **The realized-variance robustness line is dropped,** because it is built from 5-minute bars by definition. It was reported, not decisive.
- **The notebook's STEP 2 and STEP 3 use no Drive.** STEP 3 now runs the volatility protocol only, since the portfolio protocol has been scored.

## What protects the test

The reserved stage runs the validation gate again on these rows before scoring, so the baseline is still checked on the sample the test uses. This change touches only which stock-days enter, in a way blind to PE. The portfolio's reserved results were visible when it was made. They cannot bear on a data-cleaning rule this small, but they are disclosed.
